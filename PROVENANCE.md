# Provenance and scope

This project began from the [ADS 2026 Spring course project](https://github.com/DyingCoderLin/ADS-26-spring-project). The course supplied the initial inference framework, TODO-marked exercises, tests, explanatory materials, and the chatbox interface.

My coursework implementation covers:

- Qwen3.5 dynamic cache updates for full-attention K/V and linear-attention convolution/recurrent state.
- Prefill/decode cache plumbing and token-by-token decode.
- Prefix-cache snapshot cloning, longest-prefix matching, insertion, and reuse.
- CPU state serialization, SSD storage, in-memory LRU eviction, reload, and persistent index recovery.
- Context-message assembly, rolling summary compression, and session save/load.

The small `chat_app.py` CLI flags for SSD cache configuration were added while preparing this standalone repository. The web UI and surrounding engine scaffold were course-provided and are not presented as wholly independent work.

This is a new repository with a clean history rather than a copy of the course Git history. It excludes downloaded Qwen model files, student-ID submission packages, screenshots, virtual environments, and generated user conversations. No license is asserted for the course-provided code.
