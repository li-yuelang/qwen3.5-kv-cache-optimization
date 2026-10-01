from __future__ import annotations

import json
import time
from typing import Any, Iterable, Optional

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from .config import GenerationConfig
from .decoding import decode_stream, decode_tokens
from .prefix_cache import PrefixCache, TieredPrefixCache
from .tools import parse_tool_calls


def parse_messages(messages_json: str) -> list[dict[str, Any]]:
    if not messages_json:
        return []
    data = json.loads(messages_json)
    if isinstance(data, list):
        return data
    raise ValueError("messages must be a JSON list.")


class TinyQwenEngine:
    def __init__(
        self,
        model_name_or_path: str,
        enable_prefix_cache: bool = False,
        prefix_cache_entries: int = 32,
        # ── Phase 4：SSD Offloading ──────────────────────────────────────────
        # 传入一个目录路径即启用两级 Prefix Cache（内存 + SSD）；None 则维持
        # Phase 3 行为（纯内存 PrefixCache）。`prefix_cache_mem_entries` 控制
        # 两级缓存内存层的容量——故意设得比 `prefix_cache_entries` 小，用来
        # 演示 offload 到 SSD 的效果。
        prefix_cache_ssd_dir: str | None = None,
        prefix_cache_mem_entries: int = 4,
    ):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.config._attn_implementation = "eager"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Phase 3 / 4：Prefix Cache
        #   - enable_prefix_cache=False：完全关闭（phase2 行为）
        #   - enable_prefix_cache=True 且 prefix_cache_ssd_dir=None：纯内存 PrefixCache（phase3）
        #   - enable_prefix_cache=True 且 prefix_cache_ssd_dir 给定：两级 TieredPrefixCache（phase4）
        self.prefix_cache: PrefixCache | TieredPrefixCache | None
        if not enable_prefix_cache:
            self.prefix_cache = None
        elif prefix_cache_ssd_dir is None:
            self.prefix_cache = PrefixCache(max_entries=prefix_cache_entries)
        else:
            self.prefix_cache = TieredPrefixCache(
                ssd_cache_dir=prefix_cache_ssd_dir,
                device=self.device,
                max_mem_entries=prefix_cache_mem_entries,
            )

    def _prepare_inputs(
        self,
        prompt: Optional[str],
        messages: Optional[list[dict[str, Any]]],
        enable_thinking: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if messages:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        else:
            encoded = self.tokenizer(
                prompt or "",
                return_tensors="pt",
            )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        return input_ids, attention_mask

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        gen_config: Optional[GenerationConfig] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        benchmark: bool = False,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        gen_config = gen_config or GenerationConfig()
        input_ids, attention_mask = self._prepare_inputs(prompt, messages, gen_config.enable_thinking)

        start = time.time()
        generated_ids, full_ids, timing = decode_tokens(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gen_config=gen_config,
            use_cache=use_cache,
            prefix_cache=self.prefix_cache,
        )
        end = time.time()

        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        tool_calls = parse_tool_calls(text)

        usage = {
            "prompt_tokens": int(input_ids.shape[1]),
            "completion_tokens": int(len(generated_ids)),
            "total_tokens": int(full_ids.shape[1]),
        }

        result: dict[str, Any] = {
            "text": text,
            "tool_calls": tool_calls,
            "usage": usage,
        }

        if benchmark:
            elapsed = max(end - start, 1e-6)
            prefill_s = timing["prefill_s"]
            decode_s = timing["decode_s"]
            decode_tok = timing["decode_tokens"]
            prompt_tok = timing["prompt_tokens"]
            result["metrics"] = {
                "elapsed_s": elapsed,
                "tokens_per_s": len(generated_ids) / elapsed,
                "prefill_s": prefill_s,
                "prefill_tokens_per_s": prompt_tok / max(prefill_s, 1e-6),
                "decode_s": decode_s,
                "decode_tokens_per_s": decode_tok / max(decode_s, 1e-6),
                "prefix_hit_tokens": timing.get("prefix_hit_tokens", 0),
            }
        return result

    def generate_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list[dict[str, Any]]] = None,
        gen_config: Optional[GenerationConfig] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        use_cache: bool = True,
    ) -> Iterable[str]:
        gen_config = gen_config or GenerationConfig()
        input_ids, attention_mask = self._prepare_inputs(prompt, messages, gen_config.enable_thinking)
        yield from decode_stream(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gen_config=gen_config,
            use_cache=use_cache,
            prefix_cache=self.prefix_cache,
        )
