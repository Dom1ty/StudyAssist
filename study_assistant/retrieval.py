from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document

from study_assistant import config


class CourseRetriever:
    def __init__(
        self,
        embedding_model,
        persist_directory: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        config.ensure_data_dirs()
        self.vector_store = Chroma(
            collection_name=collection_name or config.COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=str(persist_directory or config.CHROMA_DIR),
        )

    def add_documents(self, documents: Iterable[Document], ids: list[str]) -> None:
        self.vector_store.add_documents(list(documents), ids=ids)

    def query(
        self,
        query_text: str,
        course_id: str,
        k: int = config.TOP_K,
        preferred_material_types: list[str] | None = None,
    ) -> list[Document]:
        preferred = preferred_material_types or []
        try:
            results = self.vector_store.similarity_search_with_score(
                query_text,
                k=max(k * 3, 10),
                filter={"course_id": course_id},
            )
        except Exception:
            return []

        ranked = []
        for doc, score in results:
            metadata = doc.metadata or {}
            if metadata.get("course_id") != course_id:
                continue
            material_type = metadata.get("material_type", "")
            preference_rank = 0 if material_type in preferred else 1
            ranked.append((preference_rank, score, doc))

        ranked.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in ranked[:k]]
