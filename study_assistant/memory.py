from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from study_assistant import config
from study_assistant.models import MistakePattern


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", value.lower()) if len(token) > 2}


class MistakeMemoryStore:
    def __init__(self, storage_path: Path | None = None) -> None:
        config.ensure_data_dirs()
        self.storage_path = storage_path or config.MISTAKES_PATH
        self._lock = threading.Lock()
        if not self.storage_path.exists():
            self._save([])

    def _load(self) -> list[dict]:
        with self._lock:
            if not self.storage_path.exists():
                return []
            return json.loads(self.storage_path.read_text(encoding="utf-8"))

    def _save(self, data: list[dict]) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.storage_path.parent) as handle:
                json.dump(data, handle, indent=2, ensure_ascii=True)
                temp_path = Path(handle.name)
            temp_path.replace(self.storage_path)

    def list_patterns(self, course_id: str) -> list[MistakePattern]:
        items = [MistakePattern.from_dict(item) for item in self._load()]
        return [item for item in items if item.course_id == course_id]

    def find_relevant(self, course_id: str, query: str, limit: int = 3) -> list[MistakePattern]:
        query_tokens = _tokenize(query)
        ranked: list[tuple[int, int, MistakePattern]] = []
        for pattern in self.list_patterns(course_id):
            haystack = f"{pattern.concept} {pattern.mistake_label} {pattern.explanation}"
            overlap = len(query_tokens & _tokenize(haystack))
            ranked.append((overlap, pattern.recurrence_count, pattern))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked[:limit] if item[0] > 0 or item[2].recurrence_count > 1]

    def upsert(
        self,
        course_id: str,
        concept: str,
        mistake_label: str,
        explanation: str,
        evidence: str,
    ) -> MistakePattern | None:
        if not concept.strip() or not mistake_label.strip():
            return None
        if mistake_label.strip().lower() in {"none", "n/a", "no misconception identified"}:
            return None

        rows = self._load()
        now = _utc_now()
        normalized_concept = _normalize(concept)
        normalized_label = _normalize(mistake_label)

        for row in rows:
            if (
                row["course_id"] == course_id
                and _normalize(row["concept"]) == normalized_concept
                and _normalize(row["mistake_label"]) == normalized_label
            ):
                row["recurrence_count"] += 1
                row["explanation"] = explanation.strip() or row["explanation"]
                row["evidence"] = evidence.strip() or row["evidence"]
                row["updated_at"] = now
                self._save(rows)
                return MistakePattern.from_dict(row)

        pattern = MistakePattern(
            pattern_id=f"mistake-{uuid4().hex[:10]}",
            course_id=course_id,
            concept=concept.strip(),
            mistake_label=mistake_label.strip(),
            explanation=explanation.strip(),
            evidence=evidence.strip(),
            recurrence_count=1,
            created_at=now,
            updated_at=now,
        )
        rows.append(pattern.to_dict())
        self._save(rows)
        return pattern
