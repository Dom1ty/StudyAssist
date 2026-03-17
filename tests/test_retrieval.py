from __future__ import annotations

from langchain_core.documents import Document

from study_assistant.ingest import MaterialIngestor


class LoaderForContent:
    def __init__(self, docs):
        self.docs = docs

    def load(self):
        return self.docs


def test_retrieval_filters_to_selected_course(tmp_path):
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding

    from study_assistant.memory import MistakeMemoryStore
    from study_assistant.repositories import CourseRepository
    from study_assistant.retrieval import CourseRetriever

    course_repo = CourseRepository(state_path=tmp_path / "state.json")
    retriever = CourseRetriever(
        embedding_model=DeterministicFakeEmbedding(size=32),
        persist_directory=tmp_path / "chroma",
        collection_name="retrieval-filter-test",
    )
    course_a = course_repo.create_course("Linear Algebra")
    course_b = course_repo.create_course("Probability")

    ingestor_a = MaterialIngestor(
        course_repository=course_repo,
        retriever=retriever,
        uploads_dir=tmp_path / "uploads-a",
        pdf_loader_factory=lambda path: LoaderForContent(
            [Document(page_content="Matrices have eigenvalues and eigenvectors.", metadata={"page": 0})]
        ),
    )
    ingestor_b = MaterialIngestor(
        course_repository=course_repo,
        retriever=retriever,
        uploads_dir=tmp_path / "uploads-b",
        pdf_loader_factory=lambda path: LoaderForContent(
            [Document(page_content="Random variables define probability distributions.", metadata={"page": 0})]
        ),
    )

    ingestor_a.ingest_pdf(b"a", "lin-alg.pdf", course_a.course_id, "lecture_notes")
    ingestor_b.ingest_pdf(b"b", "probability.pdf", course_b.course_id, "lecture_notes")

    docs = retriever.query("eigenvalues", course_a.course_id, k=3)

    assert docs
    assert all(doc.metadata["course_id"] == course_a.course_id for doc in docs)
    assert all("Random variables" not in doc.page_content for doc in docs)
