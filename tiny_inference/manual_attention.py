from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def qwen3_5_attention_forward(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None, # Qwen3_5DynamicCache 对象，由 qwen3_5_text_forward 传入，利用它更新缓存；实现KV Cache中需用到此参数
    layer_idx: int = 0, # 当前层在模型中的索引，用于定位该层的缓存槽位；实现KV Cache中需用到此参数
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_module.head_dim)

    query_states, gate = torch.chunk(
        attn_module.q_proj(hidden_states).view(*input_shape, -1, attn_module.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = attn_module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = attn_module.k_norm(attn_module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # ===== TODO: KV Cache - Full Attention 缓存更新 (START) =====
    # 将本步产生的 K、V 追加到缓存，取回包含所有历史 token 的完整 K/V，
    # 之后的 attention 计算基于这个完整序列（而非仅本步的 K/V）。
    # 缓存不存在时（no-cache 模式），直接使用本步 K/V 即可。
    if past_key_values is not None:
        # 使用 KV Cache，将当前 K、V 追加到缓存并获取完整历史 K、V
        key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)
    # 缓存不存在时，直接使用本步 K/V
    # ===== TODO: KV Cache - Full Attention 缓存更新 (END) =====

    attn_output, attn_weights = eager_attention_forward(
        attn_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=attn_module.scaling,
        dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = attn_module.o_proj(attn_output)
    return attn_output, attn_weights
