import pytest

from triage_agent.data import tokenize


def test_store_finds_customer_by_id(store):
    assert store.get_customer("C-002").plan == "enterprise"


def test_store_returns_none_for_unknown_customer(store):
    assert store.get_customer("C-999") is None


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Error 500! Can't login.") == ["error", "500", "can", "t", "login"]


def test_tokenize_drops_thai_text():
    assert tokenize("ระบบเข้าไม่ได้ error 500") == ["error", "500"]


@pytest.mark.parametrize(
    ("query", "expected_top_id"),
    [
        ("duplicate charge refund", "KB-003"),  # ticket 1
        ("pending charges bank", "KB-002"),  # ticket 1
        ("status page region outage", "KB-005"),  # ticket 2
        ("dark mode macos system default", "KB-008"),  # ticket 3
        ("feature request schedule", "KB-009"),  # ticket 3
    ],
)
def test_kb_ranks_expected_article_first(kb, query, expected_top_id):
    article, _score = kb.search(query, top_k=3)[0]
    assert article.article_id == expected_top_id


def test_kb_respects_top_k(kb):
    assert len(kb.search("charge", top_k=1)) == 1


def test_kb_returns_nothing_for_unrelated_or_empty_query(kb):
    assert kb.search("banana smoothie recipe", top_k=3) == []
    assert kb.search("   ", top_k=3) == []
