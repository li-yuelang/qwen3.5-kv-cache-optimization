from dataclasses import dataclass


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    do_sample: bool = True
    enable_thinking: bool = False
