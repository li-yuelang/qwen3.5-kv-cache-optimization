from __future__ import annotations

import json
import re
from typing import Any


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    results: list[dict[str, Any]] = []
    for match in _TOOL_CALL_PATTERN.findall(text):
        payload = match.strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            results.append(parsed)
        elif isinstance(parsed, list):
            results.extend(item for item in parsed if isinstance(item, dict))
    return results
