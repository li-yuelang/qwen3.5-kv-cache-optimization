from __future__ import annotations

import torch
from transformers.masking_utils import create_causal_mask

from .cache import Qwen3_5DynamicCache
from .manual_attention import qwen3_5_attention_forward
from .manual_linear import qwen3_5_linear_attn_forward


@torch.no_grad()
def qwen3_5_text_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Qwen3_5DynamicCache | None,
    use_cache: bool,
) -> tuple[torch.Tensor, Qwen3_5DynamicCache | None]:
    text_model = model.model
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    inputs_embeds = text_model.embed_tokens(input_ids)

    # ===== TODO: KV Cache 初始化 - 创建缓存对象与 cache_position (START) =====
    # 实现提示：首次进入时，创建缓存对象，供后续传递
    # cache_position作用：记录当前batch每个 token 在完整序列中的绝对位置索引，后续用于 RoPE（位置编码）和 create_causal_mask（构建正确的因果掩码）
    # 生成cache_position：从已缓存序列长度开始，依次递增，直到当前batch的最后一个token
    # 使用torch.arange生成，形状为(seq_len,)的1D张量
    if use_cache:
        if past_key_values is None:
            past_key_values = Qwen3_5DynamicCache(text_model.config)
            past_seen_tokens = 0
        else:
            past_seen_tokens = past_key_values.get_seq_length()
    else:
        past_seen_tokens = 0

    seq_len = inputs_embeds.shape[1]

    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + seq_len,
        device=inputs_embeds.device,
    )

    # ===== TODO: KV Cache - 创建缓存对象与 cache_position (END) =====

    position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]

    causal_mask = create_causal_mask(
        config=text_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position, # phase1中传入None，phase2中实现功能
        past_key_values=past_key_values, # phase1中默认传入None，phase2中实现功能
        position_ids=text_position_ids, # phase1中传入None，phase2中实现功能
    )
    linear_attn_mask = text_model._update_linear_attn_mask(attention_mask, cache_position)

    hidden_states = inputs_embeds
    position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(
        text_model.layers[: text_model.config.num_hidden_layers]
    ):
        layer_mask = (
            linear_attn_mask
            if decoder_layer.layer_type == "linear_attention"
            else causal_mask
        )
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)

        if decoder_layer.layer_type == "linear_attention":
            # 对比 phase1，多将缓存对象、当前位置、层索引传入，函数内部根据缓存状态决定走 prefill 还是 decode 路径
            hidden_states = qwen3_5_linear_attn_forward(
                decoder_layer.linear_attn,
                hidden_states=hidden_states,
                attention_mask=layer_mask,
                cache_params=past_key_values,
                cache_position=cache_position,
                layer_idx=layer_idx,
            )
        else:
            # 对比 phase1，多将缓存对象和层索引传入，函数内部会将本步 K/V 追加到缓存并使用完整历史 K/V 计算注意力
            hidden_states, _ = qwen3_5_attention_forward(
                decoder_layer.self_attn,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                layer_idx=layer_idx,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = text_model.norm(hidden_states)
    logits = model.lm_head(hidden_states)

    return logits, past_key_values if use_cache else None
