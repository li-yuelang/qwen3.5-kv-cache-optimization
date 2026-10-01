"""
Phase 3：Prefix Cache

在 Phase 2 里，KV Cache 解决了**单次请求内部**「每 decode 一步都要重算整段历史」
的问题。但当我们面对**多次请求**时，仍然有大量重复计算：例如同一个系统提示词会
在每次对话里重复，多轮对话里每一轮都要对「之前的所有轮次」重新 prefill 一遍。

Prefix Cache 的核心想法：把过去请求 prefill 结束时的缓存快照**跨请求**保存下来；
当新请求到来时，如果它的 prompt 以某个已缓存 prompt 作为前缀，就直接加载那份快照，
只对「新 prompt 相对老 prompt 多出来的后缀」做 prefill —— 前缀部分的计算**彻底
省掉**。

Phase 3 实现的是**整 prompt 粒度**的Prefix Cache（最简单、最常用的一种形式）：
  - 每次 prefill 结束，将 (prompt_token_ids, 缓存快照) 作为一条记录存入 PrefixCache。
  - 新请求来时，在已保存的记录里找「最长的、作为新 prompt 前缀的」那条记录，
    加载其缓存，把新 prompt 去掉已匹配前缀后的剩余 token 交给 forward。
  - 由于被复用的缓存会被后续请求继续写入，存取时都必须 clone 以避免互相污染。

淘汰策略：**LRU（Least Recently Used）**
----------------------------------------
容量有限时需要选择丢弃哪条记录。朴素的 FIFO 会把"最早插入"的记录丢掉，但
"最早"未必"最没用"——一条很早就插入、但每次请求都会命中的系统提示词，FIFO
同样会把它淘汰。LRU 改成以"最近一次被使用的时间"排序：每当一条记录被命中
（lookup）或被再次写入（insert 遇到相同 key）时，把它标记为"最近使用"；容量
溢出时丢弃"最久未使用"的那一条。这样长期被频繁命中的热点前缀会一直留在缓存里。

实现上用 `collections.OrderedDict` 维护"按最近使用时间排序"的记录：
  - 队尾（last）= 最近使用
  - 队首（first）= 最久未使用
  - 命中或复写 → `move_to_end(key)`
  - 容量溢出 → `popitem(last=False)`
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .cache import Qwen3_5DynamicCache


class PrefixCache:
    """
    最简单的Prefix Cache实现：用一个 OrderedDict 保存所有「prompt → cache 快照」
    记录，lookup 时线性扫描找最长前缀匹配，命中后把该条目移到队尾以更新 LRU 序。
    对于教学与小规模演示足够直观。真实推理引擎（如 vLLM）会用 radix tree + 块级
    哈希来支持 O(log n) 匹配与更细粒度的块级共享，核心思想是一致的。
    """

    def __init__(self, max_entries: int = 32):
        # OrderedDict 从队首到队尾按「最久未使用 → 最近使用」排列。
        # key 是 prompt_token_ids 的 tuple；value 是 cache 快照。
        self._entries: "OrderedDict[tuple[int, ...], Qwen3_5DynamicCache]" = OrderedDict()
        self._max_entries = max_entries

        # 统计信息（测试脚本会读取这两个计数器以展示命中率）
        self.hits: int = 0
        self.misses: int = 0
        self.hit_tokens: int = 0  # 累计被Prefix Cache命中而省下的 prefill token 数
        self.evictions: int = 0   # 累计因容量溢出被 LRU 淘汰的条目数

    # ---------- 查询 ----------

    def lookup(
        self, token_ids: list[int]
    ) -> tuple[int, "Qwen3_5DynamicCache | None"]:
        """
        在已保存的记录里找「最长的、作为 token_ids 前缀的」一条，返回
        (matched_len, cloned_cache)。

        约束（为简化实现与避免 forward 收到空输入）
        --------------------------------------------
        1. 若没有任何记录满足「该条记录的 tokens 是 token_ids 的前缀」，返回 (0, None)。
        2. 若找到的 matched_len == len(token_ids)（新 prompt 和某条记录完全相等），
           则把 matched_len 截断为 len(token_ids) - 1 —— forward 至少需要 1 个新 token
           来产生本次请求的首个输出 logits。
        3. 命中时，为避免快照被后续请求污染，**必须返回 cache 的 clone 而非原对象**。
        4. **LRU 维护**：命中后必须把命中的那条记录 `move_to_end`，把它标记为「最近
           使用」，这样后续 insert 触发淘汰时不会被丢掉。

        实现提示
        --------
        1. 维护一个 best_key = None, best_len = 0，遍历 self._entries.items()，
           对每一条 (cached_tokens, cached_cache)：
             - 用 tuple 切片判断 cached_tokens 是否等于 token_ids 的前 len(cached_tokens) 项；
             - 若是，则它是一个合法前缀；比较长度，择最长者。
        2. 未命中：self.misses += 1，返回 (0, None)。
        3. 命中：
             - 按约束 2 处理完全匹配的情况；
             - self._entries.move_to_end(best_key) 更新 LRU 顺序；
             - self.hits += 1, self.hit_tokens += matched_len；
             - 返回 (matched_len, self._entries[best_key].clone())。
        """
        # ===== TODO: Prefix Cache - (START) =====
        best_key = None
        best_len = 0

        for cached_tokens, cached_cache in self._entries.items():
            cached_len = len(cached_tokens)
            if cached_len <= len(token_ids) and cached_tokens == tuple(token_ids[:cached_len]):
                if cached_len > best_len:
                    best_key = cached_tokens
                    best_len = cached_len

        if best_len == 0:
            self.misses += 1
            return (0, None)

        if best_len == len(token_ids):
            best_len = len(token_ids) - 1

        self._entries.move_to_end(best_key)
        self.hits += 1
        self.hit_tokens += best_len

        return (best_len, self._entries[best_key].clone())
        # ===== TODO: Prefix Cache - (END) =====

    # ---------- 插入 ----------

    def insert(
        self, token_ids: list[int], cache: "Qwen3_5DynamicCache"
    ) -> None:
        """
        在一次 prefill 完成后，把 (prompt tokens, 当前缓存的深拷贝) 追加到记录中。

        约束
        ----
        1. 若已存在一条记录其 tokens 与 token_ids 完全相等，**不重新 clone**，但
           要把它 `move_to_end` 以更新 LRU 顺序（这条前缀刚刚又被用过一次）。
        2. 若 self._entries 已达 self._max_entries，按 LRU 淘汰"最久未使用"的一条：
           `self._entries.popitem(last=False)`，并把 self.evictions 加 1。
        3. 存入的必须是 cache.clone() —— 否则后续 decode 向该 cache 追加 K/V 时
           会把本快照一起"带着"往前走。

        实现提示
        --------
        1. 先把 token_ids 转成 tuple（list 不可 hash、也不便作键）。
        2. 若 key 已在 self._entries 中：move_to_end(key) 后 return。
        3. 若 len(self._entries) >= self._max_entries：popitem(last=False)，
           self.evictions += 1。
        4. self._entries[key] = cache.clone()（新插入的条目天然位于队尾 = 最近使用）。
        """
        # ===== TODO: Prefix Cache - (START) =====
        key = tuple(token_ids)

        if key in self._entries:
            self._entries.move_to_end(key)
            return

        if len(self._entries) >= self._max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

        self._entries[key] = cache.clone()
        # ===== TODO: Prefix Cache - (END) =====

    # ---------- 辅助 ----------

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """重置所有记录与统计信息。"""
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0
        self.evictions = 0


# ============================================================================
# Phase 4：Prefix Cache 向 SSD 的 Offloading
# ============================================================================
#
# 背景
# ----
# Phase 3 的 PrefixCache 把所有快照都放在 GPU / CPU 的**内存**里。现实推理服务
# 中，内存容量有限：Qwen3.5-0.8B 一条长 prompt 的缓存就已达到几十 MB，数百条
# 以上就会撑爆显存。生产系统（vLLM 的 CPU offload、SGLang 的 HiCache、
# DeepSeek 的 Context Caching）都会把热数据留在 GPU、把冷数据**下沉到 SSD**。
#
# Phase 4 要做的就是这件事：
#
#   ┌─────────────────┐   满了就按 LRU 驱逐   ┌─────────────────┐
#   │  Memory (hot)   │ ────────────────────→ │   SSD (cold)    │
#   │  OrderedDict    │                       │  torch.save()   │
#   │  max_entries=N  │ ←──── lookup ──────── │  目录:*.pt 文件 │
#   └─────────────────┘                       └─────────────────┘
#         ↑                                           ↑
#     命中立即返回                              命中则 load 回内存
#                                              （若内存满先驱逐再 load）
#
# 设计要点
# --------
# 1. **两级存储**：TieredPrefixCache 维护一个 mem 级（OrderedDict，LRU）和一个
#    SSD 级（DiskStore）。lookup 先查 mem，miss 再查 SSD；命中 SSD 时把该条
#    promote 到 mem。
# 2. **LRU 驱逐**：mem 满时把"最久未使用"的条目 offload 到 SSD（而非直接丢弃），
#    这样只有彻底超出 SSD 容量时才会真正丢数据。
# 3. **进程间持久化**：SSD 上的文件以 prompt tokens 的 hash 命名，启动时自动扫盘
#    重建索引。这意味着重启进程、甚至换一个 Python 解释器，命中过的 prompt
#    仍然不用重新 prefill。
# 4. **跨 device 兼容**：落盘前统一 `.cpu()`，加载时 `.to(device)`——这使得
#    GPU 训练、CPU 推理、换卡等情形都能正确恢复。
# ============================================================================


def _hash_tokens(token_ids: tuple[int, ...]) -> str:
    """把 prompt tokens 映射到 40 字节的 hex 文件名。sha1 碰撞概率可忽略，
    且同一个 prompt 无论被哪个进程落盘，都会命中同一个文件名——这正是
    "进程间持久化"依赖的性质。"""
    h = hashlib.sha1()
    for t in token_ids:
        h.update(int(t).to_bytes(4, "little", signed=False))
    return h.hexdigest()


class DiskStore:
    """
    把 Prefix Cache 条目落盘到 SSD 的最简实现。
    目录结构：
        <root>/
            <sha1>.pt          # torch.save({"tokens": [...], "state": {...}})

    索引 self._index: dict[tuple[int, ...], str]  —— key 是 prompt tokens，
    value 是对应文件的绝对路径。索引只存在于内存中，**不落盘**，因此每次
    进程启动都要通过 `_rebuild_index` 扫目录重新构建。
    """

    def __init__(self, root_dir: str, device: torch.device):
        self.root_dir = os.path.abspath(root_dir)
        self.device = device
        os.makedirs(self.root_dir, exist_ok=True)

        # key = prompt tokens tuple; value = 绝对文件路径
        self._index: dict[tuple[int, ...], str] = {}

        # 统计
        self.disk_writes: int = 0
        self.disk_reads: int = 0
        self.bytes_written: int = 0

        # 启动时扫盘重建索引（这是进程间持久化的关键步骤）
        self._rebuild_index()

    # ---------- 保存 ----------

    def save(self, token_ids: tuple[int, ...], cache: "Qwen3_5DynamicCache") -> str:
        """
        把一条 (tokens, cache) 以 `torch.save` 方式写到 SSD。

        文件内容约定（一个 dict）
        -------------------------
            {
                "tokens": list[int],         # 原始 prompt tokens，用于扫盘时重建 key
                "state":  dict[str, Any],    # cache.to_cpu_state_dict() 的返回值
            }

        实现提示
        --------
        1. 用 `_hash_tokens(token_ids)` 生成文件名，拼成绝对路径 `path`。
        2. 若 `token_ids` 已在 self._index 且文件存在，直接 return 原路径
           （内容相同，省一次写盘）。
        3. 调 cache.to_cpu_state_dict() 得到可落盘的 state。
        4. payload = {"tokens": list(token_ids), "state": state}；torch.save(payload, path)。
        5. 更新 self._index[token_ids] = path；self.disk_writes += 1；
           self.bytes_written += os.path.getsize(path)。
        6. 返回 path。
        """
        # ===== TODO: SSD Offload - DiskStore.save (START) =====
        path = os.path.join(self.root_dir, _hash_tokens(token_ids) + ".pt")

        if token_ids in self._index and os.path.exists(path):
            return path

        state = cache.to_cpu_state_dict()
        payload = {"tokens": list(token_ids), "state": state}
        torch.save(payload, path)

        self._index[token_ids] = path
        self.disk_writes += 1
        self.bytes_written += os.path.getsize(path)

        return path
        # ===== TODO: SSD Offload - DiskStore.save (END) =====

    # ---------- 加载 ----------

    def load(self, token_ids: tuple[int, ...]) -> "Qwen3_5DynamicCache | None":
        """
        从 SSD 中按 tokens 加载一条记录，返回重建好的 cache（已搬到 self.device）。
        若索引中没有该 key，或文件被外部删掉了，返回 None。

        实现提示
        --------
        1. path = self._index.get(token_ids)；若 None 或 !os.path.exists(path)，
           （如有必要）pop 掉脏索引，return None。
        2. payload = torch.load(path, map_location="cpu", weights_only=False)。
           （weights_only=False：我们存的是带 Python 容器的 dict，不是纯权重。）
        3. 从 .cache 模块 `import Qwen3_5DynamicCache`（放在函数内以避免循环 import），
           调用 `Qwen3_5DynamicCache.from_cpu_state_dict(payload["state"], self.device)`。
        4. self.disk_reads += 1；返回重建的 cache。
        """
        # ===== TODO: SSD Offload - DiskStore.load (START) =====
        path = self._index.get(token_ids)
        if path is None or not os.path.exists(path):
            self._index.pop(token_ids, None)
            return None

        payload = torch.load(path, map_location="cpu", weights_only=False)

        from .cache import Qwen3_5DynamicCache
        cache = Qwen3_5DynamicCache.from_cpu_state_dict(payload["state"], self.device)

        self.disk_reads += 1
        return cache
        # ===== TODO: SSD Offload - DiskStore.load (END) =====

    # ---------- 扫盘重建索引（进程间持久化的关键） ----------

    def _rebuild_index(self) -> None:
        """
        遍历 self.root_dir 下的所有 `*.pt` 文件，读出每个文件里的 `tokens` 字段，
        恢复 self._index，使 "上个进程写入的快照" 能在本进程被 lookup 命中。

        实现提示
        --------
        1. for fn in os.listdir(self.root_dir)：
             - 跳过非 `.pt` 文件；
             - path = os.path.join(self.root_dir, fn)；
             - 尝试 torch.load(path, map_location="cpu", weights_only=False)；
               任何异常都 try/except 住、warning 后 continue——坏文件不应让
               整个引擎起不来。
             - key = tuple(payload["tokens"])；self._index[key] = path。
        2. 不要在这里把 state 也 load 进来——那会吃爆内存。我们只读索引。
        """
        # ===== TODO: SSD Offload - DiskStore 扫盘重建索引 (START) =====
        for fn in os.listdir(self.root_dir):
            if not fn.endswith(".pt"):
                continue

            path = os.path.join(self.root_dir, fn)
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                key = tuple(payload["tokens"])
                self._index[key] = path
            except Exception:
                import warnings
                warnings.warn(f"Failed to load {path}, skipping...")
                continue
        # ===== TODO: SSD Offload - DiskStore 扫盘重建索引 (END) =====

    # ---------- 辅助 ----------

    def __contains__(self, token_ids: tuple[int, ...]) -> bool:
        return token_ids in self._index

    def __len__(self) -> int:
        return len(self._index)

    def keys(self):
        return self._index.keys()

    def delete(self, token_ids: tuple[int, ...]) -> None:
        """删除一条 SSD 记录（用于测试/清理；正常路径不需要调用）。"""
        path = self._index.pop(token_ids, None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def clear(self) -> None:
        """清空 SSD 目录与索引（测试用）。"""
        for path in list(self._index.values()):
            try:
                os.remove(path)
            except OSError:
                pass
        self._index.clear()
        self.disk_writes = 0
        self.disk_reads = 0
        self.bytes_written = 0


class TieredPrefixCache:
    """
    两级 Prefix Cache：**内存 LRU** + **SSD 持久化**。对外接口与 PrefixCache 完全
    一致（`lookup(token_ids) -> (matched_len, cache|None)` 与
    `insert(token_ids, cache) -> None`），因此可以直接替换 PrefixCache 喂给
    manual_decoding，无需改动其他代码。

    内存层
    ------
    OrderedDict，语义与 Phase 3 的 PrefixCache 完全相同：队尾 = 最近使用，
    `max_mem_entries` 限额。区别在于"溢出时不再直接丢，而是 offload 到 SSD"。

    SSD 层
    ------
    DiskStore，无条目数上限（只受磁盘空间限制——教学演示够用；真实系统会加）。
    命中 SSD 时把条目 promote 回内存；若内存也满了，先 LRU 驱逐一条到 SSD。
    """

    def __init__(
        self,
        ssd_cache_dir: str,
        device: torch.device,
        max_mem_entries: int = 4,
    ):
        self._mem: "OrderedDict[tuple[int, ...], Qwen3_5DynamicCache]" = OrderedDict()
        self._max_mem_entries = max_mem_entries
        self._disk = DiskStore(ssd_cache_dir, device)

        # 统计
        self.hits: int = 0
        self.misses: int = 0
        self.hit_tokens: int = 0
        self.mem_hits: int = 0         # 命中内存
        self.ssd_hits: int = 0         # 命中 SSD（包括启动后首次从 SSD 恢复）
        self.offloads: int = 0         # 内存→SSD 的驱逐/下沉次数

    # ---------- 查询 ----------

    def lookup(
        self, token_ids: list[int]
    ) -> tuple[int, "Qwen3_5DynamicCache | None"]:
        """
        先在内存中找最长前缀；再到 SSD 中找最长前缀；取两者中更长的一条返回。
        命中 SSD 时，把该条 promote 到内存（内存满先 LRU 驱逐到 SSD）。

        约束（与 Phase 3 一致）
        ----------------------
        1. 未命中 → misses += 1，返回 (0, None)。
        2. 完全匹配时 matched_len 截断为 len(token_ids) - 1。
        3. 返回 cache 必须是 clone，不能让外部拿到内部引用。
        4. 命中内存 → move_to_end 更新 LRU；命中 SSD → promote（见 `_promote_to_mem`）。

        实现提示
        --------
        1. 扫 self._mem.items()，找最长前缀 best_mem_key / best_mem_len。
        2. 扫 self._disk.keys()，找最长前缀 best_ssd_key / best_ssd_len。
           （keys 返回 tuple[int, ...]，用 tuple 切片判前缀即可。）
        3. 若两者均无，misses += 1，返回 (0, None)。
        4. 取更长者为 winner。若 winner 在内存：
             - self._mem.move_to_end(winner_key)
             - cache_to_clone = self._mem[winner_key]
             - self.mem_hits += 1
           否则（winner 在 SSD）：
             - cache_to_clone = self._promote_to_mem(winner_key)
             - self.ssd_hits += 1
        5. 按完全匹配截断 matched_len；self.hits += 1；self.hit_tokens += matched_len；
           返回 (matched_len, cache_to_clone.clone())。

        > 小提示：`_promote_to_mem` 已实现好，直接调用即可。
        """
        # ===== TODO: SSD Offload - TieredPrefixCache.lookup (START) =====
        best_mem_key = None
        best_mem_len = 0

        for key, cache in self._mem.items():
            key_len = len(key)
            if key_len <= len(token_ids) and key == tuple(token_ids[:key_len]):
                if key_len > best_mem_len:
                    best_mem_key = key
                    best_mem_len = key_len

        best_ssd_key = None
        best_ssd_len = 0

        for key in self._disk.keys():
            key_len = len(key)
            if key_len <= len(token_ids) and key == tuple(token_ids[:key_len]):
                if key_len > best_ssd_len:
                    best_ssd_key = key
                    best_ssd_len = key_len

        if best_mem_len == 0 and best_ssd_len == 0:
            self.misses += 1
            return (0, None)

        if best_mem_len >= best_ssd_len:
            winner_key = best_mem_key
            winner_len = best_mem_len
            self._mem.move_to_end(winner_key)
            cache_to_clone = self._mem[winner_key]
            self.mem_hits += 1
        else:
            winner_key = best_ssd_key
            winner_len = best_ssd_len
            cache_to_clone = self._promote_to_mem(winner_key)
            self.ssd_hits += 1

        if winner_len == len(token_ids):
            winner_len = len(token_ids) - 1

        self.hits += 1
        self.hit_tokens += winner_len

        return (winner_len, cache_to_clone.clone())
        # ===== TODO: SSD Offload - TieredPrefixCache.lookup (END) =====

    # ---------- 插入 ----------

    def insert(
        self, token_ids: list[int], cache: "Qwen3_5DynamicCache"
    ) -> None:
        """
        Prefill 结束后调用。把 (tokens, cache.clone()) 放到**内存**层，并维护 LRU：
        内存满了就把"最久未使用"的条目 offload 到 SSD。

        约束
        ----
        1. 若 key 已在内存：move_to_end 后 return（无需重复 clone）。
        2. 若 key 不在内存但已在 SSD：说明它刚被 promote 过、或别的进程写过。
           **仍需写内存**以完成本次插入；SSD 那份留着不动即可（内容一致）。
        3. 若内存满（len >= max_mem_entries）：
             - evict_key, evict_cache = self._mem.popitem(last=False)
             - self._disk.save(evict_key, evict_cache)
             - self.offloads += 1
        4. self._mem[key] = cache.clone()  （新条目天然位于队尾）。

        实现提示
        --------
        - token_ids 转 tuple 后作 key。
        - 1/2/3/4 按上述顺序实现即可。
        """
        # ===== TODO: SSD Offload - TieredPrefixCache.insert (START) =====
        key = tuple(token_ids)

        if key in self._mem:
            self._mem.move_to_end(key)
            return

        if len(self._mem) >= self._max_mem_entries:
            evict_key, evict_cache = self._mem.popitem(last=False)
            self._disk.save(evict_key, evict_cache)
            self.offloads += 1

        self._mem[key] = cache.clone()
        # ===== TODO: SSD Offload - TieredPrefixCache.insert (END) =====

    # ---------- 辅助（已实现） ----------

    def _promote_to_mem(self, key: tuple[int, ...]) -> "Qwen3_5DynamicCache":
        """
        把 SSD 上的一条记录加载回内存，并放在队尾（最近使用）。
        若内存已满，先按 LRU 把队首驱逐回 SSD（注意：被驱逐的条目本来也可能在
        SSD 里，DiskStore.save 会跳过重复写）。

        学生无需修改，但需要读懂——lookup 会调用它。
        """
        if len(self._mem) >= self._max_mem_entries:
            evict_key, evict_cache = self._mem.popitem(last=False)
            self._disk.save(evict_key, evict_cache)
            self.offloads += 1

        loaded = self._disk.load(key)
        if loaded is None:
            # 理论上不会发生（lookup 里已确认 key 在 SSD）。兜底返回，避免崩溃。
            raise RuntimeError(f"DiskStore lost entry for key len={len(key)}")
        self._mem[key] = loaded  # 放到队尾 = 最近使用
        return loaded

    def __len__(self) -> int:
        return len(self._mem) + len(self._disk)

    def clear(self) -> None:
        """重置内存层 + SSD 层 + 所有统计。测试脚本用。"""
        self._mem.clear()
        self._disk.clear()
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0
        self.mem_hits = 0
        self.ssd_hits = 0
        self.offloads = 0
