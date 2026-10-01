from __future__ import annotations

import torch

from .config import GenerationConfig


def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return logits
    return logits / temperature


def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p is None or top_p <= 0 or top_p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    sorted_logits = torch.where(sorted_mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
    original = torch.full_like(logits, float("-inf"))
    return original.scatter(-1, sorted_indices, sorted_logits)


def sample_next_token(logits: torch.Tensor, gen_config: GenerationConfig) -> torch.Tensor:
    if not gen_config.do_sample or gen_config.temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    filtered = _apply_temperature(logits, gen_config.temperature)
    filtered = _top_k_filter(filtered, gen_config.top_k)
    filtered = _top_p_filter(filtered, gen_config.top_p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)
