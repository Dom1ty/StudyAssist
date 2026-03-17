from __future__ import annotations

from study_assistant.memory import MistakeMemoryStore


def test_mistake_store_upserts_similar_patterns(tmp_path):
    store = MistakeMemoryStore(storage_path=tmp_path / "mistakes.json")

    first = store.upsert(
        course_id="course-1",
        concept="Chain Rule",
        mistake_label="Forgot inner derivative",
        explanation="The derivative of the inside function is missing.",
        evidence="d/dx sin(x^2) = cos(x^2)",
    )
    second = store.upsert(
        course_id="course-1",
        concept="chain   rule",
        mistake_label="forgot inner derivative",
        explanation="Still missing the derivative of the inside expression.",
        evidence="d/dx cos(3x) = -sin(3x)",
    )

    assert first is not None
    assert second is not None
    assert second.recurrence_count == 2
    patterns = store.list_patterns("course-1")
    assert len(patterns) == 1
