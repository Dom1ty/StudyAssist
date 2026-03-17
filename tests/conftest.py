from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from study_assistant.ingest import MaterialIngestor
from study_assistant.memory import MistakeMemoryStore
from study_assistant.repositories import CourseRepository
from study_assistant.retrieval import CourseRetriever


class FakePDFLoader:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def load(self) -> list[Document]:
        return self._docs


@pytest.fixture
def fake_loader_factory():
    def factory(file_path: str):
        return FakePDFLoader(
            [
                Document(page_content="Gradient descent updates parameters using the learning rate.", metadata={"page": 0}),
                Document(page_content="Convex functions have a single global minimum in this setting.", metadata={"page": 1}),
            ]
        )

    return factory


@pytest.fixture
def services(tmp_path: Path, fake_loader_factory):
    course_repo = CourseRepository(state_path=tmp_path / "state.json")
    embeddings = DeterministicFakeEmbedding(size=32)
    retriever = CourseRetriever(
        embedding_model=embeddings,
        persist_directory=tmp_path / "chroma",
        collection_name="test-study-assistant",
    )
    ingestor = MaterialIngestor(
        course_repository=course_repo,
        retriever=retriever,
        uploads_dir=tmp_path / "uploads",
        pdf_loader_factory=fake_loader_factory,
    )
    mistake_store = MistakeMemoryStore(storage_path=tmp_path / "mistakes.json")
    return {
        "course_repo": course_repo,
        "retriever": retriever,
        "ingestor": ingestor,
        "mistake_store": mistake_store,
    }
