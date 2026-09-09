"""Small, auditable answer extraction for numeric reasoning datasets."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


def _last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    start += len(marker)
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def extract_answer(text: str) -> str | None:
    """Extract the last boxed answer, with a numeric fallback."""

    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "")
    cleaned = cleaned.removeprefix("$").removesuffix("$").strip()
    cleaned = cleaned.replace(r"\,", "")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized in {"-0", "+0", ""} else normalized


def numeric_answer_is_correct(candidate: str | None, reference: str) -> bool:
    candidate_value = normalize_numeric_answer(candidate)
    reference_value = normalize_numeric_answer(reference)
    return candidate_value is not None and candidate_value == reference_value


__all__ = [
    "extract_answer",
    "normalize_numeric_answer",
    "numeric_answer_is_correct",
]
