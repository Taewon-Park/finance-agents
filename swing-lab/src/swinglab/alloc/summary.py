"""리밸런싱 문서의 LLM 요약 (선택). 로컬 LLM(Ollama 호환 /api/generate)만 쓴다.

LLM은 숫자를 만들지 않는다. 요약에 나온 숫자가 문서(표)에 없는 숫자면 요약을 버린다.
숫자 비교: 쉼표·부호·% 를 떼고 값으로 비교한다 (1.50 = 1.5, "−3.2%p"는 3.2로도 맞는 것으로 본다).
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from swinglab.alloc.config import LlmConfig

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

PROMPT = """아래는 이번 분기 리밸런싱 문서다. 이번 분기에 무엇이 왜 바뀌는지 3~5문장으로 한국어로 설명하라.
규칙:
- 새로운 숫자를 만들지 마라. 숫자를 쓰려면 문서에 있는 숫자를 그대로 써라.
- 매매를 권유하거나 전망하지 말고 문서의 내용만 설명하라.
- 문장만 쓰고 표나 목록은 쓰지 마라.

{document}
"""


def numbers_in(text: str) -> set[Decimal]:
    out = set()
    for token in _NUMBER.findall(text):
        try:
            out.add(abs(Decimal(token.replace(",", ""))).normalize())
        except InvalidOperation:
            continue
    return out


def unknown_numbers(summary: str, document: str) -> list[str]:
    """요약에 있지만 문서에 없는 숫자 (빈 목록이면 통과)."""
    allowed = numbers_in(document)
    return sorted(str(n) for n in numbers_in(summary) - allowed)


def checked_summary(summary: str | None, document: str) -> tuple[str | None, str]:
    """(통과한 요약 또는 None, 버린 이유)."""
    if not summary or not summary.strip():
        return None, "LLM 응답이 비어 있음"
    bad = unknown_numbers(summary, document)
    if bad:
        return None, f"요약에 표에 없는 숫자가 있어 버림 ({', '.join(bad)})"
    return summary.strip(), ""


Opener = Callable[..., object]


def request_summary(document: str, cfg: LlmConfig, opener: Opener = urllib.request.urlopen) -> tuple[str | None, str]:
    """로컬 LLM 호출. (응답 또는 None, 실패 이유). 타임아웃·재시도 후에도 실패하면 None."""
    body = json.dumps({"model": cfg.model, "prompt": PROMPT.format(document=document), "stream": False}).encode()
    last_error = ""
    for attempt in range(1, cfg.max_retries + 1):
        req = urllib.request.Request(cfg.endpoint, data=body, headers={"Content-Type": "application/json"})
        try:
            with opener(req, timeout=cfg.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload.get("response")
            if isinstance(text, str):
                return text, ""
            last_error = "응답에 response 필드가 없음"
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < cfg.max_retries:
            logger.warning("LLM 요약 실패 (%d/%d): %s", attempt, cfg.max_retries, last_error)
            time.sleep(1.0)
    return None, f"LLM 호출 실패 ({last_error})"
