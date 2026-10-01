from __future__ import annotations

import time

import torch

from .config import GenerationConfig
from .manual_qwen3_5 import qwen3_5_text_forward
from .prefix_cache import PrefixCache
from .sampling import sample_next_token


def _eos_token_ids(model) -> set[int]:
    """Match HF GenerationMixin: stop if *any* configured EOS id is sampled."""
    raw = getattr(model.config, "eos_token_id", None)
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple)):
        return {int(x) for x in raw if x is not None}
    return {int(raw)}


@torch.no_grad()
def decode_tokens_manual(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
    use_cache: bool = True,
    prefix_cache: PrefixCache | None = None,
) -> tuple[list[int], torch.Tensor, dict[str, float]]:
    model.eval()
    generated: list[int] = []
    eos_ids = _eos_token_ids(model)

    # ===== TODO: Prefix Cache - (START) =====
    # 在走到 prefill 之前，先让 prefix_cache 出手看看能不能省掉一部分 prefill。
    # 当 use_cache=True 且 prefix_cache 不为 None 时：
    #   1. 调用 prefix_cache.lookup(input_ids[0].tolist())，得到
    #      (matched_len, loaded_cache)。
    #   2. 若 matched_len > 0：
    #        - 把 prefill 要真正跑的输入截断为 input_ids[:, matched_len:]
    #          （前缀对应的 K/V 已经在 loaded_cache 中，forward 会把剩余后缀
    #          追加到缓存末尾）；attention_mask 不变，它仍覆盖完整序列长度，
    #          forward 内部会基于缓存长度正确计算 cache_position。
    #        - 把 loaded_cache 作为 past_key_values 传给 forward。
    #   3. 若 matched_len == 0：保持 prefill_input_ids = input_ids, past_kv = None。
    # 没有开启 Prefix Cache 时：prefill_input_ids = input_ids, past_kv = None。
    #
    # 命中时要记录省下了多少 prefill token（matched_len），后面写入 timing。
    prefill_input_ids = input_ids
    prefill_past_kv = None
    prefix_hit_tokens = 0
    if use_cache and prefix_cache is not None:
        matched_len, loaded_cache = prefix_cache.lookup(input_ids[0].tolist())
        if matched_len > 0:
            prefill_input_ids = input_ids[:, matched_len:]
            prefill_past_kv = loaded_cache
            prefix_hit_tokens = matched_len
    # ===== TODO: Prefix Cache - (END) =====


    prefill_start = time.time()
    logits, past_key_values = qwen3_5_text_forward(
        model=model,
        input_ids=prefill_input_ids,
        attention_mask=attention_mask,
        past_key_values=prefill_past_kv,
        use_cache=use_cache,
    )
    prefill_end = time.time()

    # ===== TODO: Prefix Cache - (START) =====
    # Prefill 刚结束时，past_key_values 恰好等于「整段 prompt 的缓存状态」，
    # 此刻是为Prefix Cache拍快照的最佳时机（再晚一步就会被 decode 污染）。
    # 当 use_cache=True 且 prefix_cache 不为 None 时：
    #   调用 prefix_cache.insert(input_ids[0].tolist(), past_key_values)
    # 注意：insert 内部会负责 clone，这里不用手动 clone。
    if use_cache and prefix_cache is not None:
        prefix_cache.insert(input_ids[0].tolist(), past_key_values)
    # ===== TODO: Prefix Cache - (END) =====

    # 从 prefill 输出的最后一个位置采样第一个新 token
    logits = logits[:, -1, :]
    next_token = sample_next_token(logits, gen_config)
    token_id = int(next_token.item())
    generated.append(token_id)

    attention_mask = torch.cat(
        [attention_mask, torch.ones_like(next_token)], dim=1
    )

    decode_start = time.time()
    for step in range(1, gen_config.max_new_tokens):
        if token_id in eos_ids:
            break

        if use_cache:
            # ===== TODO: KV Cache - Decode 单步（用缓存替代全序列重计算）(START) =====
            # 调用 qwen3_5_text_forward，只传入本步新生成的 1 个 token（而非完整序列），
            # 同时传入 past_key_values（已有缓存）和 use_cache=True。
            # 函数返回新的 logits 和更新后的 past_key_values，供下一步继续使用。
            if next_token.dim() == 1:
                next_token = next_token.unsqueeze(1)  # 调整形状为 (1, 1)
            # 使用完整的 attention_mask，而不是仅包含当前 token 的 mask
            logits, past_key_values = qwen3_5_text_forward(
                model=model,
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            
            # ===== TODO: KV Cache - Decode 单步（用缓存替代全序列重计算）(END) =====
        else:
            full_ids = torch.cat(
                [input_ids, torch.tensor([generated], device=input_ids.device)], dim=1
            )
            logits, _ = qwen3_5_text_forward(
                model=model,
                input_ids=full_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
            )

        logits = logits[:, -1, :]
        next_token = sample_next_token(logits, gen_config)
        token_id = int(next_token.item())
        generated.append(token_id)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )
    decode_end = time.time()

    full_ids = torch.cat(
        [input_ids, torch.tensor([generated], device=input_ids.device)], dim=1
    )

    timing = {
        "prefill_s": prefill_end - prefill_start,
        "decode_s": decode_end - decode_start,
        "decode_tokens": max(len(generated) - 1, 1),
        "prompt_tokens": int(input_ids.shape[1]),
        "prefix_hit_tokens": prefix_hit_tokens,
    }
    return generated, full_ids, timing


@torch.no_grad()
def decode_stream_manual(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
    use_cache: bool = True,
    prefix_cache: PrefixCache | None = None,
):
    model.eval()
    eos_ids = _eos_token_ids(model)

    # 流式解码同样支持Prefix Cache：lookup → partial prefill → insert
    # 具体逻辑参考 decode_tokens_manual 中的两个 Prefix Cache TODO 块；此处按相同方式处理。
    prefill_input_ids = input_ids
    prefill_past_kv = None
    prefix_hit_tokens = 0
    if use_cache and prefix_cache is not None:
        matched_len, loaded_cache = prefix_cache.lookup(input_ids[0].tolist())
        if matched_len > 0:
            prefill_input_ids = input_ids[:, matched_len:]
            prefill_past_kv = loaded_cache
            prefix_hit_tokens = matched_len

    # 将 prompt（或其去前缀后的后缀）喂入模型，forward 内部自动创建/延续所有层的缓存
    prefill_start = time.time()
    logits, past_key_values = qwen3_5_text_forward(
        model=model,
        input_ids=prefill_input_ids,
        attention_mask=attention_mask,
        past_key_values=prefill_past_kv,
        use_cache=use_cache,
    )
    prefill_end = time.time()

    if use_cache and prefix_cache is not None:
        prefix_cache.insert(input_ids[0].tolist(), past_key_values)

    logits = logits[:, -1, :]
    next_token = sample_next_token(logits, gen_config)
    token_id = int(next_token.item())
    generated: list[int] = [token_id]
    attention_mask = torch.cat(
        [attention_mask, torch.ones_like(next_token)], dim=1
    )

    if token_id not in eos_ids:
        piece = tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if piece:
            yield piece

    decode_start = time.time()
    decode_count = 0
    for step in range(1, gen_config.max_new_tokens):
        if token_id in eos_ids:
            break

        if use_cache:
            # ===== TODO: KV Cache - Streaming Decode 单步（用缓存替代全序列重计算）(START) =====
            # 启用缓存时：只输入刚生成的 1 个 token，携带已有缓存，forward 内部追加并返回更新后的缓存
            if next_token.dim() == 1:
                next_token = next_token.unsqueeze(1)  # 调整形状为 (1, 1)
            # 使用完整的 attention_mask，而不是仅包含当前 token 的 mask
            logits, past_key_values = qwen3_5_text_forward(
                model=model,
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            
            # ===== TODO: KV Cache - Streaming Decode 单步（用缓存替代全序列重计算）(END) =====
        else:
            full_ids = torch.cat(
                [input_ids, torch.tensor([generated], device=input_ids.device)], dim=1
            )
            logits, _ = qwen3_5_text_forward(
                model=model,
                input_ids=full_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
            )

        logits = logits[:, -1, :]
        next_token = sample_next_token(logits, gen_config)
        token_id = int(next_token.item())
        generated.append(token_id)
        decode_count += 1
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )
        if token_id in eos_ids:
            break
        piece = tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if piece:
            yield piece
    decode_end = time.time()

    # 输出 prefill / decode 速度统计
    prefill_time = prefill_end - prefill_start
    decode_time = decode_end - decode_start
    prompt_tokens = int(input_ids.shape[1])
    decode_tokens = max(decode_count, 1)

    yield f"\n\n--- Performance ---\n"
    yield f"Prefill : {prompt_tokens} tokens in {prefill_time:.3f}s → {prompt_tokens / max(prefill_time, 1e-6):.2f} tokens/s\n"
    yield f"Decode  : {decode_tokens} tokens in {decode_time:.3f}s → {decode_tokens / max(decode_time, 1e-6):.2f} tokens/s\n"



