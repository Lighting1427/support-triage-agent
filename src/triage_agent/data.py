import json
import re
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from rank_bm25 import BM25Okapi

from triage_agent.models import (
    BillingAccount,
    ComponentStatus,
    Customer,
    KBArticle,
    TicketDataset,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tickets(path: Path) -> TicketDataset:
    return TicketDataset.model_validate(_read_json(path))


class MockDataStore:
    """Customer, billing, and service-status data queried by the tools."""

    def __init__(self, data_dir: Path) -> None:
        customers = TypeAdapter(list[Customer]).validate_python(
            _read_json(data_dir / "customers.json")
        )
        accounts = TypeAdapter(list[BillingAccount]).validate_python(
            _read_json(data_dir / "billing.json")
        )
        self._customers = {c.customer_id: c for c in customers}
        self._billing = {a.customer_id: a for a in accounts}
        self._statuses = TypeAdapter(list[ComponentStatus]).validate_python(
            _read_json(data_dir / "service_status.json")
        )

    def get_customer(self, customer_id: str) -> Customer | None:
        return self._customers.get(customer_id)

    def get_billing(self, customer_id: str) -> BillingAccount | None:
        return self._billing.get(customer_id)

    def service_statuses(self) -> list[ComponentStatus]:
        return list(self._statuses)


# ---------------------------------------------------------------------------
# Knowledge Base search
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text: 'Error 500!' -> ['error', '500']."""
    return _TOKEN_PATTERN.findall(text.lower())


class KnowledgeBase:
    """Search articles with BM25.

    BM25 is sufficient for this knowledge base of just 10 articles.
    """

    def __init__(self, articles: list[KBArticle]) -> None:
        self._articles = articles
        self._bm25 = BM25Okapi([tokenize(self._index_text(a)) for a in articles])

    @classmethod
    def from_file(cls, path: Path) -> "KnowledgeBase":
        return cls(TypeAdapter(list[KBArticle]).validate_python(_read_json(path)))

    @property
    def article_ids(self) -> frozenset[str]:
        return frozenset(a.article_id for a in self._articles)

    @staticmethod
    def _index_text(article: KBArticle) -> str:
        boosted = f"{article.title} {' '.join(article.tags)}"
        return f"{boosted} {boosted} {article.content}"

    def search(self, query: str, top_k: int) -> list[tuple[KBArticle, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        ranked = sorted(zip(self._articles, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(article, round(float(score), 3)) for article, score in ranked[:top_k] if score > 0]
