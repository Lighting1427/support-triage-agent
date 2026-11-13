import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    openai_model: str
    data_dir: Path
    recursion_limit: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            data_dir=Path(os.getenv("TRIAGE_DATA_DIR", PROJECT_ROOT / "data")),
        )