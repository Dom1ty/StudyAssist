from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from study_assistant import config
from study_assistant.ingest import MaterialIngestor
from study_assistant.llm import build_default_chat_model, build_default_embeddings
from study_assistant.memory import MistakeMemoryStore
from study_assistant.models import TutorMode, TutorResponse
from study_assistant.repositories import CourseRepository
from study_assistant.retrieval import CourseRetriever
from study_assistant.tutor import TutorService


@dataclass
class ServiceContainer:
    course_repo: CourseRepository
    mistake_store: MistakeMemoryStore
    retriever: CourseRetriever
    ingestor: MaterialIngestor
    tutor: TutorService
    using_fallback_model: bool


@st.cache_resource(show_spinner=False)
def build_services() -> ServiceContainer:
    config.ensure_data_dirs()
    course_repo = CourseRepository()
    mistake_store = MistakeMemoryStore()
    embeddings = build_default_embeddings()
    retriever = CourseRetriever(embedding_model=embeddings)
    ingestor = MaterialIngestor(course_repository=course_repo, retriever=retriever)
    chat_model = build_default_chat_model()
    tutor = TutorService(retriever=retriever, mistake_store=mistake_store, chat_model=chat_model)
    return ServiceContainer(
        course_repo=course_repo,
        mistake_store=mistake_store,
        retriever=retriever,
        ingestor=ingestor,
        tutor=tutor,
        using_fallback_model=chat_model is None,
    )


def render_answer(answer: TutorResponse) -> None:
    st.markdown(answer.plain_language_explanation or "No explanation returned.")

    if answer.concepts:
        st.markdown("**Concepts**")
        for concept in answer.concepts:
            st.write(f"- {concept}")

    if answer.prerequisites:
        st.markdown("**Prerequisites**")
        for item in answer.prerequisites:
            st.write(f"- {item}")

    if answer.steps:
        st.markdown("**Step-by-step guidance**")
        for index, step in enumerate(answer.steps, start=1):
            st.write(f"{index}. {step}")

    if answer.diagnosis:
        st.markdown("**Diagnosis**")
        st.write(answer.diagnosis)

    if answer.next_step:
        st.markdown("**Next step**")
        st.write(answer.next_step)

    if answer.confidence_note:
        st.caption(answer.confidence_note)

    if answer.sources:
        with st.expander("Supporting course context"):
            for source in answer.sources:
                page_label = f", page {source.page}" if source.page else ""
                st.markdown(f"**{source.source}** ({source.material_type}{page_label})")
                st.write(source.excerpt)


def main() -> None:
    st.set_page_config(page_title="Study Assistant", page_icon=":books:", layout="wide")
    services = build_services()
    st.title("Study Assistant")
    st.caption("A retrieval-grounded tutor that prioritizes understanding before speed.")

    if services.using_fallback_model:
        st.warning(
            "No `DASHSCOPE_API_KEY` was found. Retrieval and storage still work, but tutor responses fall back to a local heuristic mode."
        )

    courses = services.course_repo.list_courses()
    selected_course_id = services.course_repo.get_selected_course_id()
    if selected_course_id is None and courses:
        selected_course_id = courses[0].course_id
        services.course_repo.set_selected_course(selected_course_id)

    with st.sidebar:
        st.header("Courses")
        with st.form("create-course", clear_on_submit=True):
            new_course_name = st.text_input("New course name")
            create_course = st.form_submit_button("Create course")
        if create_course:
            if new_course_name.strip():
                course = services.course_repo.create_course(new_course_name)
                selected_course_id = course.course_id
                st.rerun()
            else:
                st.warning("Enter a course name first.")

        courses = services.course_repo.list_courses()
        if courses:
            course_lookup = {course.course_id: course.name for course in courses}
            default_index = 0
            if selected_course_id in course_lookup:
                default_index = list(course_lookup.keys()).index(selected_course_id)
            selected_course_id = st.selectbox(
                "Active course",
                options=list(course_lookup.keys()),
                index=default_index,
                format_func=lambda course_id: course_lookup[course_id],
            )
            services.course_repo.set_selected_course(selected_course_id)
        else:
            st.info("Create a course to begin.")

        st.divider()
        st.header("Materials")
        material_type = st.selectbox(
            "Upload as",
            options=["lecture_notes", "exercises", "solutions"],
            index=0,
        )
        uploads = st.file_uploader(
            "Upload course PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("Index uploads", disabled=not (selected_course_id and uploads)):
            for uploaded in uploads or []:
                try:
                    result = services.ingestor.ingest_pdf(
                        file_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        course_id=selected_course_id,
                        material_type=material_type,
                    )
                    if result.status == "duplicate":
                        st.info(result.message)
                    else:
                        st.success(result.message)
                except Exception as exc:
                    st.error(f"{uploaded.name}: {exc}")

        if selected_course_id:
            materials = services.course_repo.list_materials(selected_course_id)
            if materials:
                st.caption("Indexed files")
                for material in materials:
                    st.write(f"- {material.filename} [{material.material_type}] ({material.chunk_count} chunks)")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}

    history = st.session_state["chat_history"].setdefault(selected_course_id or "default", [])

    if not selected_course_id:
        st.info("Create a course, upload PDFs, and then start asking questions.")
        return

    course = services.course_repo.get_course(selected_course_id)
    st.subheader(course.name if course else "Current course")

    left, right = st.columns([2, 1], gap="large")
    with left:
        for entry in history:
            with st.chat_message("user"):
                st.markdown(entry["question"])
                if entry.get("student_attempt"):
                    st.caption("My attempt")
                    st.write(entry["student_attempt"])
            with st.chat_message("assistant"):
                render_answer(TutorResponse.from_dict(entry["answer"]))

        with st.form("ask-tutor", clear_on_submit=True):
            mode = st.radio(
                "Tutor mode",
                options=[TutorMode.HINT.value, TutorMode.GUIDED.value, TutorMode.FULL_SOLUTION.value],
                horizontal=True,
            )
            question = st.text_area("Ask about the course material", height=120)
            student_attempt = st.text_area("My attempt (optional)", height=120)
            submit = st.form_submit_button("Ask tutor")

        if submit:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                answer = services.tutor.answer(
                    course_id=selected_course_id,
                    question=question,
                    mode=TutorMode(mode),
                    student_attempt=student_attempt,
                )
                history.append(
                    {
                        "question": question,
                        "student_attempt": student_attempt,
                        "answer": answer.to_dict(),
                    }
                )
                st.rerun()

    with right:
        st.markdown("**Mistake log**")
        patterns = services.mistake_store.list_patterns(selected_course_id)
        if not patterns:
            st.write("No recurring mistakes recorded yet.")
        else:
            for pattern in sorted(patterns, key=lambda item: item.recurrence_count, reverse=True):
                with st.expander(f"{pattern.concept}: {pattern.mistake_label} ({pattern.recurrence_count})"):
                    st.write(pattern.explanation)
                    if pattern.evidence:
                        st.caption("Example evidence")
                        st.write(pattern.evidence)


if __name__ == "__main__":
    main()
