from __future__ import annotations

import json
import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from study_assistant import config
from study_assistant.memory import MistakeMemoryStore
from study_assistant.models import SourceSnippet, TutorMode, TutorResponse
from study_assistant.retrieval import CourseRetriever


TUTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a patient university tutor. Prioritize understanding, reasoning, and correctness. "
            "Explain the idea before the procedure. Point out misconceptions explicitly and in plain language. "
            "Do not optimize for exams over understanding. "
            "Return valid JSON only with these keys: "
            "response_type, plain_language_explanation, concepts, prerequisites, steps, diagnosis, mistake_label, next_step, confidence_note. "
            "concepts, prerequisites, and steps must be arrays of short strings.",
        ),
        (
            "human",
            "Tutor mode: {mode}\n"
            "Turn type: {turn_type}\n"
            "Question: {question}\n"
            "Student attempt: {student_attempt}\n"
            "Relevant course context:\n{context}\n\n"
            "Prior mistakes for this course:\n{prior_mistakes}\n\n"
            "Mode rules:\n"
            "- Hint: provide at most 2 steps and avoid giving away the full solution.\n"
            "- Guided: explain the concept first, then give step-by-step guidance.\n"
            "- Full solution: provide the complete worked reasoning after the simple explanation.\n"
            "- If the student attempt is wrong, diagnose why and whether the diagnosis is inference-based.\n",
        ),
    ]
)


class TutorService:
    def __init__(
        self,
        retriever: CourseRetriever,
        mistake_store: MistakeMemoryStore,
        chat_model=None,
    ) -> None:
        self.retriever = retriever
        self.mistake_store = mistake_store
        self.chat_model = chat_model

    def answer(
        self,
        course_id: str,
        question: str,
        mode: TutorMode = TutorMode.GUIDED,
        student_attempt: str = "",
    ) -> TutorResponse:
        turn_type = self.classify_turn(question, student_attempt)
        query_text = f"{question}\n{student_attempt}".strip()
        docs = self.retriever.query(
            query_text=query_text,
            course_id=course_id,
            k=config.TOP_K,
            preferred_material_types=self._preferred_material_types(turn_type),
        )
        prior_mistakes = self.mistake_store.find_relevant(course_id, query_text)
        response = self._generate_response(
            question=question,
            student_attempt=student_attempt,
            mode=mode,
            turn_type=turn_type,
            docs=docs,
            prior_mistakes=prior_mistakes,
        )
        response.sources = self._build_sources(docs)
        self._remember_mistake(course_id, response, student_attempt)
        return response

    @staticmethod
    def classify_turn(question: str, student_attempt: str = "") -> str:
        text = question.lower()
        if student_attempt.strip():
            return "error_diagnosis"
        concept_markers = ("what is", "why", "explain", "intuition", "understand", "concept")
        problem_markers = ("solve", "calculate", "compute", "derive", "prove", "find")
        if any(marker in text for marker in concept_markers):
            return "concept_explanation"
        if any(marker in text for marker in problem_markers):
            return "problem_solving"
        return "mixed"

    def _preferred_material_types(self, turn_type: str) -> list[str]:
        if turn_type == "concept_explanation":
            return ["lecture_notes", "exercises"]
        if turn_type == "problem_solving":
            return ["exercises", "lecture_notes", "solutions"]
        if turn_type == "error_diagnosis":
            return ["solutions", "exercises", "lecture_notes"]
        return ["lecture_notes", "exercises", "solutions"]

    def _generate_response(
        self,
        question: str,
        student_attempt: str,
        mode: TutorMode,
        turn_type: str,
        docs: list[Document],
        prior_mistakes: Iterable,
    ) -> TutorResponse:
        context = self._format_context(docs)
        prior_mistakes_text = self._format_prior_mistakes(prior_mistakes)

        if self.chat_model is None:
            return self._fallback_response(question, student_attempt, mode, turn_type, docs, prior_mistakes_text)

        try:
            prompt_value = TUTOR_PROMPT.invoke(
                {
                    "mode": mode.value,
                    "turn_type": turn_type,
                    "question": question,
                    "student_attempt": student_attempt or "No student attempt provided.",
                    "context": context,
                    "prior_mistakes": prior_mistakes_text,
                }
            )
            result = self.chat_model.invoke(prompt_value.to_messages())
            content = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_json_response(content)
            return TutorResponse.from_dict(parsed)
        except Exception:
            return self._fallback_response(question, student_attempt, mode, turn_type, docs, prior_mistakes_text)

    def _fallback_response(
        self,
        question: str,
        student_attempt: str,
        mode: TutorMode,
        turn_type: str,
        docs: list[Document],
        prior_mistakes_text: str,
    ) -> TutorResponse:
        context_summary = docs[0].page_content if docs else "I do not have matching course context yet."
        simple_explanation = (
            "Plain-language idea: "
            + self._trim_text(context_summary, 320)
            + (" Start by relating the question back to the core concept." if docs else " Upload course material to ground the explanation.")
        )
        concepts = self._extract_keywords(question)[:3]
        if not concepts and docs:
            concepts = self._extract_keywords(docs[0].page_content)[:3]
        prerequisites = concepts[:1]
        steps_by_mode = {
            TutorMode.HINT: [
                "Identify which concept from the notes matches the question before computing anything.",
                "Write the next equation or rule you would apply, but stop before the final answer.",
            ],
            TutorMode.GUIDED: [
                "State the relevant concept in your own words.",
                "Map the known information from the problem to that concept.",
                "Carry out one justified step at a time and check each step against the concept.",
            ],
            TutorMode.FULL_SOLUTION: [
                "State the relevant concept in plain language.",
                "Translate the problem into the right formal setup.",
                "Execute the derivation carefully and justify each transformation.",
                "Check whether the result matches the original question and assumptions.",
            ],
        }

        diagnosis = ""
        mistake_label = ""
        confidence_note = ""
        if student_attempt.strip():
            diagnosis = (
                "Your attempt should be checked against the underlying concept, not only the final answer. "
                "Based on the available course context, the likely issue is a mismatch between the rule you applied and the problem setup."
            )
            mistake_label = "Possible concept-application mismatch"
            if not any(doc.metadata.get("material_type") == "solutions" for doc in docs):
                confidence_note = "Diagnosis is inference-based because no matching solution was retrieved."

        if prior_mistakes_text != "No prior mistake patterns.":
            confidence_note = (confidence_note + " Prior course mistakes were considered.").strip()

        return TutorResponse(
            response_type=turn_type,
            plain_language_explanation=simple_explanation,
            concepts=concepts,
            prerequisites=prerequisites,
            steps=steps_by_mode[mode],
            diagnosis=diagnosis,
            mistake_label=mistake_label,
            next_step="Try the next step yourself, then compare it with the concept and the retrieved notes.",
            confidence_note=confidence_note,
            sources=[],
        )

    def _remember_mistake(self, course_id: str, response: TutorResponse, student_attempt: str) -> None:
        if not student_attempt.strip():
            return
        self.mistake_store.upsert(
            course_id=course_id,
            concept=response.concepts[0] if response.concepts else "Unspecified concept",
            mistake_label=response.mistake_label or "",
            explanation=response.diagnosis or "",
            evidence=student_attempt,
        )

    def _format_context(self, docs: list[Document]) -> str:
        if not docs:
            return "No relevant course context was retrieved."
        lines = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "?")
            material_type = doc.metadata.get("material_type", "unknown")
            excerpt = self._trim_text(doc.page_content, 500)
            lines.append(f"[{index}] {source} (page {page}, {material_type}): {excerpt}")
        return "\n".join(lines)

    def _format_prior_mistakes(self, patterns: Iterable) -> str:
        rows = []
        for pattern in patterns:
            rows.append(
                f"- Concept: {pattern.concept}; mistake: {pattern.mistake_label}; explanation: {self._trim_text(pattern.explanation, 180)}"
            )
        return "\n".join(rows) if rows else "No prior mistake patterns."

    def _build_sources(self, docs: list[Document]) -> list[SourceSnippet]:
        seen: set[tuple[str, int | None]] = set()
        sources: list[SourceSnippet] = []
        for doc in docs:
            source = str(doc.metadata.get("source", "Unknown source"))
            page = doc.metadata.get("page")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceSnippet(
                    source=source,
                    page=page if isinstance(page, int) else None,
                    material_type=str(doc.metadata.get("material_type", "")),
                    excerpt=self._trim_text(doc.page_content, 350),
                )
            )
        return sources

    def _extract_keywords(self, text: str) -> list[str]:
        stopwords = {
            "about",
            "after",
            "before",
            "their",
            "there",
            "which",
            "would",
            "could",
            "should",
            "explain",
            "solve",
            "question",
            "course",
        }
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
        unique: list[str] = []
        for word in words:
            lowered = word.lower()
            if lowered in stopwords or lowered in unique:
                continue
            unique.append(lowered)
        return unique

    def _parse_json_response(self, content: str) -> dict:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("Model output was not a JSON object.")
        return payload

    @staticmethod
    def _trim_text(text: str, limit: int) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."
