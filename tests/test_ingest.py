from __future__ import annotations


def test_ingest_pdf_assigns_course_and_page_metadata(services):
    course = services["course_repo"].create_course("Machine Learning")
    result = services["ingestor"].ingest_pdf(
        file_bytes=b"%PDF-test",
        filename="lecture1.pdf",
        course_id=course.course_id,
        material_type="lecture_notes",
    )

    assert result.status == "indexed"
    assert result.indexed_chunk_count >= 2

    docs = services["retriever"].query("learning rate", course.course_id, k=5)
    assert docs
    for doc in docs:
        assert doc.metadata["course_id"] == course.course_id
        assert doc.metadata["material_type"] == "lecture_notes"
        assert doc.metadata["source"] == "lecture1.pdf"
        assert isinstance(doc.metadata["page"], int)
        assert "chunk_id" in doc.metadata
