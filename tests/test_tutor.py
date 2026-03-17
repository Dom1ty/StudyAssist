from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from study_assistant.memory import MistakeMemoryStore
from study_assistant.models import TutorMode
from study_assistant.tutor import TutorService


class StaticRetriever:
    def query(self, query_text, course_id, k, preferred_material_types):
        return [
            Document(
                page_content="A derivative measures the local rate of change of a function.",
                metadata={"source": "notes.pdf", "page": 2, "material_type": "lecture_notes", "course_id": course_id},
            )
        ]


def test_tutor_response_preserves_plain_explanation_and_guided_steps(tmp_path):
    payload = {
        "response_type": "problem_solving",
        "plain_language_explanation": "A derivative tells you how fast the function is changing right now.",
        "concepts": ["derivative"],
        "prerequisites": ["limits"],
        "steps": [
            "Identify the function and the variable.",
            "Apply the derivative rule that matches the function.",
            "Simplify the result and interpret it in context.",
        ],
        "diagnosis": "",
        "mistake_label": "",
        "next_step": "Differentiate one similar example on your own.",
        "confidence_note": "",
    }
    tutor = TutorService(
        retriever=StaticRetriever(),
        mistake_store=MistakeMemoryStore(storage_path=tmp_path / "mistakes.json"),
        chat_model=FakeListChatModel(responses=[json.dumps(payload)]),
    )

    answer = tutor.answer(
        course_id="course-1",
        question="Explain how to solve this derivative problem",
        mode=TutorMode.GUIDED,
    )

    assert answer.plain_language_explanation.startswith("A derivative tells you")
    assert answer.steps[0] == "Identify the function and the variable."
    assert answer.response_type == "problem_solving"
    assert answer.sources[0].source == "notes.pdf"
