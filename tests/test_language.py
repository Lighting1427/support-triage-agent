"""Deterministic language detection."""

from triage_agent.language import contains_thai, detect_reply_language


def test_contains_thai():
    assert contains_thai("ระบบเข้าไม่ได้ครับ error 500")
    assert not contains_thai("Payment failed $29.99 🙂")


def test_reply_language_follows_what_the_customer_wrote():
    assert detect_reply_language(["Hello", "ช่วยเช็คให้หน่อยครับ"]) == "th"
    assert detect_reply_language(["Hello", "Is this a bug?"]) == "en"
