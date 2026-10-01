"""
Qwen3.5 动态缓存：同时管理 Full Attention 的 KV Cache 和 Linear Attention 的 conv/recurrent state。
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class Qwen3_5DynamicCache:
    """
    混合动态缓存，为每一层维护：
      - Full Attention 层：key_cache / value_cache（沿 seq_len 维度拼接增长）
      - Linear Attention 层：conv_states（卷积滑动窗口）/ recurrent_states（记忆矩阵 S）
    """

    def __init__(self, config):
        # 各层类型列表，元素为 "full_attention" 或 "linear_attention"
        # 在 qwen3_5_text_forward 的层循环中用于判断走哪条 forward 路径
        self.layer_types = config.layer_types

        # full_attention(即transformer_layer) 层的索引列表
        # 例如 [3, 7, 11, ...]，表示第一个full_attention层的索引为3，第二个full_attention层的索引为7，以此类推。
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers)
            if self.layer_types[i] == "full_attention"
        ]

        # 最后一个 linear_attention 层的索引
        # 用于 has_previous_state 函数判断：prefill 完成后该层的 conv_state 会被写入，以此作为「prefill 已完成」的信号
        self.last_linear_layer = (
            len(self.layer_types) - 1
            - self.layer_types[::-1].index("linear_attention")
        )

        num_layers = config.num_hidden_layers

        # ── Full Attention 层的 KV Cache（延迟初始化，首次 update 时赋值）──────────────
        # key_cache[i] 形状：(batch, num_heads, seq_len, head_dim)
        #   - 在 eager_attention_forward 里作为完整历史 K 与当前 q 做点积
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers

        # value_cache[i] 形状同 key_cache[i]，用作完整历史 V
        #   - 与 key_cache 同步更新，同步拼接
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers

        # ── Linear Attention 层的两种状态（延迟初始化）────────────────────────────────

        # conv_states[i] 形状：(batch, conv_dim, kernel_size)，其中 kernel_size=4
        #   - 因果卷积的滑动窗口快照，保存最近 4 个时间步的中间投影（Q/K/V 拼接后）
        #   - prefill 结束时：保存序列末尾 4 帧（F.pad 补齐）
        #   - decode 每步：由 torch_causal_conv1d_update 原地 copy_ 更新，新帧滑入右端
        self.conv_states: list[torch.Tensor | None] = [None] * num_layers

        # recurrent_states[i] 形状：(batch, num_heads, key_head_dim, value_head_dim)
        #   - Gated Delta Net 的记忆矩阵 S，承载所有历史 token 的压缩表示
        #   - prefill 结束时：保存 torch_chunk_gated_delta_rule 输出的最终状态
        #   - decode 每步：由 torch_recurrent_gated_delta_rule 按 delta rule 更新
        self.recurrent_states: list[torch.Tensor | None] = [None] * num_layers

    # ---------- Full Attention 缓存更新 ----------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Important: Full Attention 层的缓存更新函数
        将当前步的 key_states / value_states 追加到该层缓存，返回拼接后的完整 K、V。

        由 manual_attention.py 在每次 attention forward 时调用：
          - Prefill：key_states 的 seq_len = prompt 长度，首次写入缓存
          - Decode ：key_states 的 seq_len = 1，追加到已有缓存末尾

        拼接后返回的完整 K/V 会直接传给 eager_attention_forward，
        使 attention 始终在完整历史序列上计算。

        实现提示
        --------
        1. 若该层的缓存尚未初始化，直接将本步 K、V 存入缓存。                                                                                     
        2. 否则，将本步 K、V 沿序列长度维度拼接到已有缓存末尾。                                                                                   
        3. 返回拼接后的完整 K、V 供注意力计算使用。
        """
        # ===== TODO: KV Cache - Full Attention 缓存更新 (START) =====
        # first write
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

        # append
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states],
                dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states],
                dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
        # ===== TODO: KV Cache - Full Attention 缓存更新 (END) =====

    # ---------- 辅助方法 ----------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        返回 Full Attention 层已缓存的序列长度。

        由 qwen3_5_text_forward 在每次 forward 开始时调用，
        用于计算 cache_position（新 token 在完整序列中的绝对位置偏移），提示：
          - Prefill：past_seen_tokens=0，cache_position = [0, 1, ..., prompt_len-1]
          - Decode ：past_seen_tokens=prompt_len+已生成数，cache_position = [past_seen_tokens]

        实现提示
        --------
        1. 若传入的层不是 full attention 层，则借用第一个 full attention 层的缓存来查询长度。                                                    
        2. 若该层缓存尚未初始化，返回 0。                                                                                                        
        3. 否则返回已缓存的序列长度。     
        """
        # ===== TODO: KV Cache - 返回已缓存序列长度 (START) =====
        # 检查传入的层是否为 full attention 层
        if self.layer_types[layer_idx] != "full_attention":
            # 若不是，使用第一个 full attention 层
            if len(self.transformer_layers) > 0:
                layer_idx = self.transformer_layers[0]
            else:
                return 0
        
        # 检查缓存是否初始化
        if self.key_cache[layer_idx] is None:
            return 0
        
        # 返回已缓存的序列长度（seq_len 维度是第 3 维）
        return self.key_cache[layer_idx].shape[2]
        # ===== TODO: KV Cache - 返回已缓存序列长度 (END) =====

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        供 transformers 的 create_causal_mask 调用，返回 (kv_length, kv_offset)。
        已实现，无需修改。
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    # ---------- Phase 3：Prefix Cache所需的深拷贝 ----------

    def clone(self) -> "Qwen3_5DynamicCache":
        """
        Phase 3 Prefix Cache 专用：对当前缓存进行**深拷贝**，返回一个与原缓存完全独立的副本。

        为什么需要深拷贝
        ----------------
        Prefix Cache把 prefill 结束时的缓存作为「可复用的快照」保存下来，供后续共享相同
        前缀的请求复用。但后续请求在复用时会继续向缓存里 append 新的 K/V、原地更新
        conv_state/recurrent_state —— 如果直接引用原对象，这些写入会**污染**快照，
        导致下一次复用时拿到的不再是"prefill 刚结束时"的状态，而是夹杂着上一个请求
        若干步 decode 的状态。

        因此 PrefixCache.insert 必须先 clone 再存，PrefixCache.lookup 也必须 clone
        后再交给 decoding 使用（这样同一快照可以被多次复用）。

        实现提示
        --------
        1. 创建一个新的 Qwen3_5DynamicCache 实例（可利用 __new__ 避免重新走 __init__
           所需的 config；将 layer_types / transformer_layers / last_linear_layer 拷过去即可）。
        2. 对四个 list（key_cache / value_cache / conv_states / recurrent_states）
           逐项处理：若元素为 None 则仍为 None；若为 Tensor 则使用 `.clone()` 得到独立副本。
        3. 返回新实例。
        """
        # ===== TODO: Prefix Cache - (START) =====
        new_cache = object.__new__(Qwen3_5DynamicCache)
        new_cache.layer_types = self.layer_types
        new_cache.transformer_layers = self.transformer_layers
        new_cache.last_linear_layer = self.last_linear_layer

        def _clone_list(lst):
            return [None if item is None else item.clone() for item in lst]

        new_cache.key_cache = _clone_list(self.key_cache)
        new_cache.value_cache = _clone_list(self.value_cache)
        new_cache.conv_states = _clone_list(self.conv_states)
        new_cache.recurrent_states = _clone_list(self.recurrent_states)

        return new_cache
        # ===== TODO: Prefix Cache - (END) =====

    # ---------- Phase 4：SSD Offloading 所需的序列化 / 反序列化 ----------

    def to_cpu_state_dict(self) -> dict[str, Any]:
        """
        Phase 4 SSD Offloading 专用：把当前缓存的全部 tensor 拉到 CPU 并打包成一个
        普通 dict，使其可以被 `torch.save` 落盘到 SSD。

        为什么需要"拉到 CPU"
        --------------------
        Prefix Cache 的快照原本住在 GPU / MPS 上。若直接 `torch.save` GPU tensor，
        落盘的文件里会嵌入 device 信息，跨机器、跨 GPU、或 GPU 不可用时加载会出错；
        同时频繁往磁盘写 GPU 显存也没有意义（SSD 本身就是 CPU 侧设备）。标准做法是
        在序列化前统一 `.cpu()`，反序列化后由调用方 `.to(device)` 搬回。

        返回值结构（建议）
        ------------------
        返回一个扁平 dict，形如：
          {
            "layer_types": list[str],
            "transformer_layers": list[int],
            "last_linear_layer": int,
            "key_cache":        list[Tensor|None],   # 全部 .cpu()
            "value_cache":      list[Tensor|None],
            "conv_states":      list[Tensor|None],
            "recurrent_states": list[Tensor|None],
          }

        实现提示
        --------
        1. 四个 tensor list 逐项处理：None 保持 None，Tensor 调 `.detach().cpu()`。
        2. 三个元信息字段（layer_types / transformer_layers / last_linear_layer）直接拷。
        3. 不要 clone——反正 tensor 已离开 GPU，且 `torch.save` 会再做一次拷贝。
        """
        # ===== TODO: SSD Offload - cache 序列化 (START) =====
        return {
            "layer_types": self.layer_types,
            "transformer_layers": self.transformer_layers,
            "last_linear_layer": self.last_linear_layer,
            "key_cache": [None if t is None else t.detach().cpu() for t in self.key_cache],
            "value_cache": [None if t is None else t.detach().cpu() for t in self.value_cache],
            "conv_states": [None if t is None else t.detach().cpu() for t in self.conv_states],
            "recurrent_states": [None if t is None else t.detach().cpu() for t in self.recurrent_states],
        }
        # ===== TODO: SSD Offload - cache 序列化 (END) =====

    @classmethod
    def from_cpu_state_dict(
        cls, state: dict[str, Any], device: torch.device
    ) -> "Qwen3_5DynamicCache":
        """
        Phase 4 SSD Offloading 专用：从 `to_cpu_state_dict` 生成的 dict 恢复出一个
        完整的 Qwen3_5DynamicCache 实例，并把所有 tensor 搬到目标 device 上。

        注意
        ----
        - 这是 `@classmethod`，不走 `__init__`（__init__ 需要 config 参数，而落盘时
          并没有保存 config）。请用 `object.__new__(cls)` 构造"空壳"，再把字段填上。
          —— 这与 `clone()` 里"跳过 __init__"的技巧相同。
        - 返回的 cache 应当与原 cache 在结构上完全一致：同样长度的四个 list，None
          位置保持 None，Tensor 位置已在 `device` 上。

        实现提示
        --------
        1. `obj = object.__new__(cls)`
        2. 拷三个元信息：layer_types / transformer_layers / last_linear_layer。
        3. 四个 tensor list 逐项处理：None→None；Tensor→`.to(device)`。
        4. 返回 obj。
        """
        # ===== TODO: SSD Offload - cache 反序列化 (START) =====
        obj = object.__new__(cls)
        obj.layer_types = state["layer_types"]
        obj.transformer_layers = state["transformer_layers"]
        obj.last_linear_layer = state["last_linear_layer"]

        def to_device_list(lst):
            return [None if t is None else t.to(device) for t in lst]

        obj.key_cache = to_device_list(state["key_cache"])
        obj.value_cache = to_device_list(state["value_cache"])
        obj.conv_states = to_device_list(state["conv_states"])
        obj.recurrent_states = to_device_list(state["recurrent_states"])

        return obj
        # ===== TODO: SSD Offload - cache 反序列化 (END) =====

    @property
    def has_previous_state(self) -> bool:
        """
        判断 prefill 是否已完成，即缓存中是否已有上一步的状态。
        原理：prefill 完成后，最后一个 linear 层的 conv_state 会被写入（非 None），
        因此可用它作为「所有层的 prefill 均已完成」的信号。

        由 qwen3_5_linear_attn_forward 在每次调用时检查，决定走 prefill 还是 decode 路径：
          - False（prefill）：使用完整 conv1d + chunk-wise delta rule
          - True （decode） ：使用单步卷积 torch_causal_conv1d_update
                             + 单步递推 torch_recurrent_gated_delta_rule

        已实现，无需修改。
        """
        return self.conv_states[self.last_linear_layer] is not None

