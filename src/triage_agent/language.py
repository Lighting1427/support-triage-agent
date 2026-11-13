"""Deterministic language checks shared by preprocessing, policy, and evaluation.

Supports only Thai and English, matching the sample data. For production support
of more languages, use a language-detection library such as lingua while keeping
the same function signatures.
"""

import re
from collections.abc import Iterable

_THAI_SCRIPT = re.compile(r"[\u0E00-\u0E7F]")  # Unicode range for Thai script.


def contains_thai(text: str) -> bool:
    return bool(_THAI_SCRIPT.search(text))


def detect_reply_language(customer_texts: Iterable[str]) -> str:
    """Return "th" if any customer text contains Thai script, otherwise "en"."""
    return "th" if any(contains_thai(text) for text in customer_texts) else "en"


def reply_matches_language(text: str, language: str) -> bool:
    if not text.strip():
        return False
    if language == "th":
        return contains_thai(text)
    if language == "en":
        return not contains_thai(text)
    return False
