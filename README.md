# Qwen3.5 KV Cache and Prefix Cache

A compact inference project for studying KV caching in Qwen3.5's hybrid attention architecture. It combines per-request cache reuse during prefill/decode, cross-request prefix reuse, memory/SSD cache tiers, and context management for multi-turn chat.

This repository is a curated copy of my work on the [ADS 2026 Spring course project](https://github.com/DyingCoderLin/ADS-26-spring-project). The course supplied the base inference framework, task scaffolding, documentation, and chatbox UI. My implementation fills in the cache, prefix-cache, SSD-offload, and context-management tasks. See [PROVENANCE.md](PROVENANCE.md) for the scope of each contribution. The model weights, course handouts, submission archives, screenshots, and local sessions are intentionally excluded.

## Five-phase roadmap

Each phase addresses a limitation exposed by the previous one. Phase 1 establishes the course-provided baseline; my implementation work is concentrated in the TODO tasks of Phases 2–5.

| Phase | Focus | What it does and why |
| --- | --- | --- |
| **1. Baseline inference** | Run Qwen3.5 without caching | Loads the model and produces a correctness/performance reference for the later optimizations. Use `--no-cache` to compare against it. |
| **2. KV/state cache** | Avoid recomputing the whole history during generation | Caches growing K/V tensors in full-attention layers and fixed-size convolution/recurrent state in linear-attention layers. Prefill runs once; decode forwards one new token at a time. |
| **3. Prefix cache** | Reuse work across requests | Finds the longest cached prompt prefix, clones its cache snapshot, and prefills only the unmatched suffix. Hit counters show how much prompt work was reused. |
| **4. Memory + SSD cache** | Keep more reusable prefixes than memory alone can hold | Uses an in-memory LRU, serializes evicted cache state to CPU/SSD, reloads it on a hit, and rebuilds the disk index across processes. |
| **5. Multi-turn context** | Keep conversations within a token budget | Compresses older turns into a rolling summary, saves/loads sessions, and assembles stable prompt prefixes to improve prefix-cache reuse. The course-provided chatbox demonstrates the result. |

## Requirements

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)
- Enough RAM/VRAM and disk space for the Qwen3.5-0.8B model; weights are **not** stored here
- Internet access on first run to download the model, or a local model directory passed with `--model`

```bash
uv sync
uv run python main.py --prompt "Explain KV Cache in one paragraph." --max-new-tokens 64 --benchmark true
```

Compare with the no-cache baseline:

```bash
uv run python main.py --prompt "Explain KV Cache in one paragraph." --max-new-tokens 64 --benchmark true --no-cache
```

Run the local chatbox with cross-request prefix caching:

```bash
uv run python chat_app.py --prefix-cache
```

Add SSD offload (the `cache_data/` directory is ignored by Git):

```bash
uv run python chat_app.py --prefix-cache --ssd-cache-dir ./cache_data --prefix-cache-mem-entries 4
```

Then open `http://127.0.0.1:8000`. The app is a teaching demo, not a production server. Model loading and long prompts can be slow on CPU. `main.py` defaults to a Hugging Face mirror when `HF_ENDPOINT` is unset; you can set `HF_ENDPOINT` yourself before running.

## Layout

- `tiny_inference/cache.py`: attention cache and state serialization
- `tiny_inference/manual_*.py`: manual attention and decode paths
- `tiny_inference/prefix_cache.py`: in-memory and SSD-backed prefix caches
- `tiny_inference/context.py`: rolling summary and persistent sessions
- `tiny_inference/engine.py`: model loading and generation API
- `main.py`: single-prompt CLI
- `chat_app.py`: local multi-turn demo UI (course-provided scaffold)

## Notes

No benchmark numbers are claimed here because results depend on hardware, prompt length, and cache hit rate. This repository does not include model weights or a license for the course starter code; consult the [upstream course project](https://github.com/DyingCoderLin/ADS-26-spring-project) and the model's license before redistributing them.
