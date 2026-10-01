from __future__ import annotations

import torch

from .config import GenerationConfig
from .manual_decoding import decode_stream_manual, decode_tokens_manual
from .prefix_cache import PrefixCache


def decode_tokens(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
    use_cache: bool = True,
    prefix_cache: PrefixCache | None = None,
) -> tuple[list[int], torch.Tensor, dict[str, float]]:
    return decode_tokens_manual(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        gen_config=gen_config,
        use_cache=use_cache,
        prefix_cache=prefix_cache,
    )


def decode_stream(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_config: GenerationConfig,
    use_cache: bool = True,
    prefix_cache: PrefixCache | None = None,
):
    yield from decode_stream_manual(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        gen_config=gen_config,
        use_cache=use_cache,
        prefix_cache=prefix_cache,
    )
