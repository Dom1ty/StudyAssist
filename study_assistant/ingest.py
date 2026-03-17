from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from study_assistant import config
from study_assistant.models import IngestResult
from study_assistant.repositories import CourseRepository
from study_assistant.retrieval import CourseRetriever


def _safe_filename(filename: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", filename).strip("._") or "upload.pdf"


def _checksum(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


class MaterialIngestor:
    def __init__(
        self,
        course_repository: CourseRepository,
        retriever: CourseRetriever,
        uploads_dir: Path | None = None,
        pdf_loader_factory: Callable[[str], object] | None = None,
        splitter: RecursiveCharacterTextSplitter | None = None,
    ) -> None:
        self.course_repository = course_repository
        self.retriever = retriever
        self.uploads_dir = uploads_dir or config.UPLOADS_DIR
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_loader_factory = pdf_loader_factory or (lambda file_path: PyPDFLoader(file_path, mode="page"))
        self.splitter = splitter or RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def ingest_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        course_id: str,
        material_type: str,
    ) -> IngestResult:
        if self.course_repository.get_course(course_id) is None:
            raise ValueError(f"Unknown course_id: {course_id}")

        checksum = _checksum(file_bytes)
        existing = self.course_repository.find_material_by_checksum(course_id, checksum)
        if existing is not None:
            return IngestResult(
                status="duplicate",
                indexed_chunk_count=existing.chunk_count,
                material=existing,
                message=f"{filename} is already indexed for this course.",
            )

        safe_name = _safe_filename(filename)
        material_stub = self.course_repository.new_material(
            course_id=course_id,
            material_type=material_type,
            filename=filename,
            stored_path="",
            checksum=checksum,
            ingest_status="processing",
            chunk_count=0,
        )
        course_upload_dir = self.uploads_dir / course_id
        course_upload_dir.mkdir(parents=True, exist_ok=True)
        stored_path = course_upload_dir / f"{material_stub.material_id}_{safe_name}"
        stored_path.write_bytes(file_bytes)

        loader = self.pdf_loader_factory(str(stored_path))
        raw_docs = loader.load()
        page_docs = self._normalize_page_docs(raw_docs, course_id, material_type, filename, material_stub.material_id, checksum)
        if not page_docs:
            raise ValueError("The uploaded PDF does not contain extractable text. Scanned PDFs are not supported in v1.")

        chunked_docs = self.splitter.split_documents(page_docs)
        if not chunked_docs:
            raise ValueError("No text chunks were produced from the uploaded PDF.")

        ids: list[str] = []
        for index, doc in enumerate(chunked_docs):
            doc.metadata["chunk_id"] = index
            ids.append(f"{material_stub.material_id}:{index}")
        self.retriever.add_documents(chunked_docs, ids=ids)

        material = self.course_repository.new_material(
            course_id=course_id,
            material_type=material_type,
            filename=filename,
            stored_path=str(stored_path),
            checksum=checksum,
            ingest_status="indexed",
            chunk_count=len(chunked_docs),
        )
        material.material_id = material_stub.material_id
        self.course_repository.add_material(material)
        return IngestResult(
            status="indexed",
            indexed_chunk_count=len(chunked_docs),
            material=material,
            message=f"Indexed {len(chunked_docs)} chunks from {filename}.",
        )

    def _normalize_page_docs(
        self,
        raw_docs: list[Document],
        course_id: str,
        material_type: str,
        filename: str,
        material_id: str,
        checksum: str,
    ) -> list[Document]:
        normalized: list[Document] = []
        for fallback_page, doc in enumerate(raw_docs, start=1):
            content = doc.page_content.strip()
            if not content:
                continue
            metadata = dict(doc.metadata or {})
            page_number = metadata.get("page", fallback_page - 1)
            try:
                page_number = int(page_number) + 1
            except (TypeError, ValueError):
                page_number = fallback_page
            normalized.append(
                Document(
                    page_content=content,
                    metadata={
                        "course_id": course_id,
                        "material_type": material_type,
                        "source": filename,
                        "page": page_number,
                        "material_id": material_id,
                        "checksum": checksum,
                    },
                )
            )
        return normalized
