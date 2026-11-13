"""ตรวจภาษาของข้อความแบบ deterministic (ใช้ทั้งใน preprocess, policy และ eval)

รองรับแค่ไทย / อังกฤษ ตามข้อมูลตัวอย่าง ถ้าต้องรองรับหลายภาษาใน production
ให้เปลี่ยนเป็น library ตรวจภาษา (เช่น lingua) โดยคง function signature เดิมไว้
"""

import re
from collections.abc import Iterable

_THAI_SCRIPT = re.compile(r"[\u0E00-\u0E7F]")  # ช่วง Unicode ของอักษรไทย


def contains_thai(text: str) -> bool:
    return bool(_THAI_SCRIPT.search(text))


def detect_reply_language(customer_texts: Iterable[str]) -> str:
    """ภาษาที่ควรใช้ตอบลูกค้า: "th" ถ้าลูกค้าเขียนภาษาไทย ไม่งั้น "en" """
    return "th" if any(contains_thai(text) for text in customer_texts) else "en"


def reply_matches_language(text: str, language: str) -> bool:
    if not text.strip():
        return False
    if language == "th":
        return contains_thai(text)
    if language == "en":
        return not contains_thai(text)
    return False
