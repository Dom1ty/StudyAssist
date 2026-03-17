from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = Path(os.getenv("STUDY_ASSISTANT_DATA_DIR", PROJECT_ROOT / "data"))
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
STATE_PATH = DATA_DIR / "state.json"
MISTAKES_PATH = DATA_DIR / "mistakes.json"

COLLECTION_NAME = os.getenv("STUDY_ASSISTANT_COLLECTION", "study_assistant_materials")
CHAT_MODEL_NAME = os.getenv("STUDY_ASSISTANT_CHAT_MODEL", "qwen3-max")
EMBEDDING_MODEL_NAME = os.getenv("STUDY_ASSISTANT_EMBEDDING_MODEL", "text-embedding-v4")

CHUNK_SIZE = int(os.getenv("STUDY_ASSISTANT_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("STUDY_ASSISTANT_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("STUDY_ASSISTANT_TOP_K", "5"))


def load_env_file(env_path: Path = ENV_PATH) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, UPLOADS_DIR, CHROMA_DIR):
        path.mkdir(parents=True, exist_ok=True)


load_env_file()
