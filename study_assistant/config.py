from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def ensure_data_dirs() -> None:
    for path in (DATA_DIR, UPLOADS_DIR, CHROMA_DIR):
        path.mkdir(parents=True, exist_ok=True)
