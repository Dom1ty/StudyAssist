from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from study_assistant import config
from study_assistant.models import Course, Material


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "course"


class CourseRepository:
    def __init__(self, state_path: Path | None = None) -> None:
        config.ensure_data_dirs()
        self.state_path = state_path or config.STATE_PATH
        self._lock = threading.Lock()
        if not self.state_path.exists():
            self._save_state({"selected_course_id": None, "courses": [], "materials": []})

    def _load_state(self) -> dict:
        with self._lock:
            if not self.state_path.exists():
                return {"selected_course_id": None, "courses": [], "materials": []}
            return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.state_path.parent) as handle:
                json.dump(state, handle, indent=2, ensure_ascii=True)
                temp_path = Path(handle.name)
            temp_path.replace(self.state_path)

    def list_courses(self) -> list[Course]:
        state = self._load_state()
        return [Course.from_dict(item) for item in state["courses"]]

    def get_course(self, course_id: str) -> Course | None:
        for course in self.list_courses():
            if course.course_id == course_id:
                return course
        return None

    def create_course(self, name: str) -> Course:
        state = self._load_state()
        now = _utc_now()
        course = Course(
            course_id=f"{_slugify(name)}-{uuid4().hex[:8]}",
            name=name.strip(),
            created_at=now,
            updated_at=now,
        )
        state["courses"].append(course.to_dict())
        state["selected_course_id"] = course.course_id
        self._save_state(state)
        return course

    def set_selected_course(self, course_id: str | None) -> None:
        state = self._load_state()
        state["selected_course_id"] = course_id
        self._save_state(state)

    def get_selected_course_id(self) -> str | None:
        return self._load_state().get("selected_course_id")

    def add_material(self, material: Material) -> Material:
        state = self._load_state()
        state["materials"].append(material.to_dict())
        for course in state["courses"]:
            if course["course_id"] == material.course_id:
                course["updated_at"] = _utc_now()
                break
        self._save_state(state)
        return material

    def list_materials(self, course_id: str | None = None) -> list[Material]:
        state = self._load_state()
        materials = [Material.from_dict(item) for item in state["materials"]]
        if course_id is None:
            return materials
        return [item for item in materials if item.course_id == course_id]

    def find_material_by_checksum(self, course_id: str, checksum: str) -> Material | None:
        for material in self.list_materials(course_id):
            if material.checksum == checksum:
                return material
        return None

    def new_material(
        self,
        course_id: str,
        material_type: str,
        filename: str,
        stored_path: str,
        checksum: str,
        ingest_status: str,
        chunk_count: int,
    ) -> Material:
        now = _utc_now()
        return Material(
            material_id=f"mat-{uuid4().hex[:10]}",
            course_id=course_id,
            material_type=material_type,
            filename=filename,
            stored_path=stored_path,
            checksum=checksum,
            ingest_status=ingest_status,
            chunk_count=chunk_count,
            created_at=now,
            updated_at=now,
        )
