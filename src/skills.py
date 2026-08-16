from __future__ import annotations

import os
import re
from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from src.config import get_path
from src.models import StudentProfile, StudentState, TeachingGoal


class TeachingSkill(BaseModel):
    model_config = ConfigDict(extra="allow")

    skill_id: str
    name: str
    version: str = "1.0"
    trigger: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    goal: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    teacher_actions: list[str] = Field(default_factory=list)
    student_signals: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    source_video: str | None = None
    source_course: str | None = None
    source_timestamp: str | None = None
    bloom_category: str | None = None
    teaching_strategies: list[str] = Field(default_factory=list)
    skill_type: str = "subject"
    added_reason: str | None = None
    applicable_when: list[str] = Field(default_factory=list)
    courses: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    required_topic_tags: list[str] = Field(default_factory=list)
    trigger_states: list[str] = Field(default_factory=list)
    required_prior_knowledge: list[str] = Field(default_factory=list)
    prerequisite_mastery_bypass: float = Field(default=0.65, ge=0.0, le=1.0)
    action_type: str = "subject_instruction"
    mastery_range: tuple[float, float] = (0.0, 1.0)
    misconception_tags: list[str] = Field(default_factory=list)

    def prompt_summary(self) -> str:
        return (
            f"{self.skill_id}｜{self.name}｜类型:{self.skill_type}｜"
            f"触发:{'；'.join(self.trigger[:3])}｜步骤:{'；'.join(self.procedure[:4])}"
        )

    def generation_summary(self) -> str:
        """Give the teacher model strategy metadata without replaying a whole lesson plan."""
        return (
            f"{self.skill_id}｜{self.name}｜类型:{self.skill_type}｜"
            f"可用动作:{'、'.join(self.teacher_actions[:5])}｜"
            f"期望信号:{'；'.join(self.student_signals[:3])}｜"
            f"适用场景:{'；'.join(self.applicable_when[:3])}"
        )


def _terms(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    ascii_words = set(re.findall(r"[a-z0-9_+-]{2,}", compact))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", compact))
    grams = {chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))}
    return ascii_words | grams


class SkillLibrary:
    def __init__(self, directory: Path | None = None, custom_directory: Path | None = None):
        self.directory = directory or get_path("skills")
        # Explicit directories remain self-contained for tests/import tools;
        # the application keeps bundled skills read-only and writes user
        # versions to a separate directory.
        self.custom_directory = custom_directory or (
            self.directory
            if directory is not None or os.getenv("TEACHING_AGENT_SKILL_DIR", "").strip()
            else get_path("skill_custom")
        )
        self.builtin_ids = {path.stem for path in self.directory.glob("*.yaml")}
        self.skills = self._load()
        self.by_id = {skill.skill_id: skill for skill in self.skills}

    def _load(self) -> list[TeachingSkill]:
        loaded: list[TeachingSkill] = []
        paths = sorted({*self.directory.glob("*.yaml"), *self.custom_directory.glob("*.yaml")}, key=lambda path: path.name)
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                data: dict[str, Any] = yaml.safe_load(handle) or {}
            loaded.append(TeachingSkill.model_validate(data))
        if not loaded:
            raise FileNotFoundError(f"Skill Library 为空: {self.directory}")
        return loaded

    def get(self, skill_id: str) -> TeachingSkill:
        if skill_id not in self.by_id:
            raise KeyError(f"Skill 不存在: {skill_id}")
        return self.by_id[skill_id]

    def by_type(self, skill_type: str) -> list[TeachingSkill]:
        return [skill for skill in self.skills if skill.skill_type == skill_type]

    @staticmethod
    def validate_import(raw: bytes | str) -> TeachingSkill:
        """Safely parse and semantically validate one uploaded Skill YAML."""
        if isinstance(raw, bytes):
            if len(raw) > 1_000_000:
                raise ValueError("Skill 文件不能超过 1MB")
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError("Skill 文件必须使用 UTF-8 编码") from exc
        else:
            text = raw
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML 格式错误：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("YAML 顶层必须是一个 Skill 对象")
        try:
            skill = TeachingSkill.model_validate(data)
        except Exception as exc:
            raise ValueError(f"Skill 字段校验失败：{exc}") from exc
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,79}", skill.skill_id):
            raise ValueError("skill_id 只能包含小写字母、数字、下划线或连字符，长度为 3–80")
        required_lists = {
            "trigger": skill.trigger,
            "goal": skill.goal,
            "procedure": skill.procedure,
            "student_signals": skill.student_signals,
            "verification": skill.verification,
        }
        missing = [name for name, values in required_lists.items() if not values]
        if missing:
            raise ValueError("以下字段必须是非空列表：" + "、".join(missing))
        if skill.skill_type != "subject" and not skill.added_reason:
            raise ValueError("新增通用 Skill 必须填写 added_reason")
        if skill.skill_type == "subject":
            structured_missing = [
                name
                for name, values in {
                    "courses": skill.courses,
                    "topic_tags": skill.topic_tags,
                    "trigger_states": skill.trigger_states,
                    "required_prior_knowledge": skill.required_prior_knowledge,
                }.items()
                if not values
            ]
            if structured_missing:
                raise ValueError("学科 Skill 缺少结构化字段：" + "、".join(structured_missing))
        return skill

    def import_skill(self, raw: bytes | str, *, new_skill_id: str | None = None) -> TeachingSkill:
        """Validate and atomically persist a Skill without overwriting an existing ID."""
        skill = self.validate_import(raw)
        if new_skill_id:
            skill.skill_id = new_skill_id.strip()
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,79}", skill.skill_id):
                raise ValueError("新的 skill_id 格式不正确")
        target = self.custom_directory / f"{skill.skill_id}.yaml"
        if target.exists() or skill.skill_id in self.by_id:
            raise FileExistsError(f"Skill ID 已存在：{skill.skill_id}")
        payload = yaml.safe_dump(
            skill.model_dump(exclude_none=True),
            allow_unicode=True,
            sort_keys=False,
        )
        self.custom_directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".tmp", prefix=f".{skill.skill_id}-", dir=self.custom_directory,
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        replace(temporary, target)
        self.skills.append(skill)
        self.skills.sort(key=lambda item: item.skill_id)
        self.by_id[skill.skill_id] = skill
        return skill

    def is_builtin(self, skill_id: str) -> bool:
        return skill_id in self.builtin_ids

    def user_skill_ids(self) -> list[str]:
        return sorted(skill.skill_id for skill in self.skills if not self.is_builtin(skill.skill_id))

    def archive_user_skill(self, skill_id: str) -> None:
        if self.is_builtin(skill_id):
            raise PermissionError("内置 Skill 只读，不能归档")
        source = self.custom_directory / f"{skill_id}.yaml"
        if not source.exists():
            raise FileNotFoundError(skill_id)
        archive = self.custom_directory / ".archive"
        archive.mkdir(exist_ok=True)
        replace(source, archive / source.name)

    def restore_user_skill(self, skill_id: str) -> None:
        source = self.custom_directory / ".archive" / f"{skill_id}.yaml"
        if not source.exists():
            raise FileNotFoundError(skill_id)
        target = self.custom_directory / source.name
        if target.exists():
            raise FileExistsError(f"Skill ID 已存在：{skill_id}")
        replace(source, target)

    def delete_user_skill(self, skill_id: str) -> None:
        if self.is_builtin(skill_id):
            raise PermissionError("内置 Skill 只读，不能删除")
        source = self.custom_directory / f"{skill_id}.yaml"
        if not source.exists():
            raise FileNotFoundError(skill_id)
        trash = self.custom_directory / ".trash"
        trash.mkdir(exist_ok=True)
        replace(source, trash / source.name)

    def list_archived_user_skills(self) -> list[str]:
        return sorted(path.stem for path in (self.custom_directory / ".archive").glob("*.yaml"))

    def rank(
        self,
        goal: TeachingGoal,
        state: StudentState,
        student_message: str = "",
        limit: int = 5,
        include_generic: bool = True,
    ) -> list[TeachingSkill]:
        query = " ".join(
            [goal.course, goal.topic, goal.objective, *goal.knowledge_points, state.next_focus, student_message]
        )
        query_terms = _terms(query)
        scored: list[tuple[float, TeachingSkill]] = []
        for skill in self.skills:
            if not include_generic and skill.skill_type != "subject":
                continue
            body = " ".join(
                [
                    skill.name,
                    skill.source_course or "",
                    *skill.trigger,
                    *skill.preconditions,
                    *skill.goal,
                    *skill.procedure,
                    *skill.applicable_when,
                ]
            )
            overlap = len(query_terms & _terms(body))
            score = float(overlap)
            if skill.source_course and (
                skill.source_course in goal.course or goal.course in skill.source_course
            ):
                score += 10
            if skill.skill_type == "subject":
                score += 1
            scored.append((score, skill))
        scored.sort(key=lambda item: (-item[0], item[1].skill_id))
        return [skill for _, skill in scored[:limit]]

    @staticmethod
    def _course_matches(skill: TeachingSkill, goal: TeachingGoal) -> bool:
        if skill.skill_type != "subject":
            return True

        def canonical(course: str) -> str:
            normalized = re.sub(r"\s+", "", course).lower()
            aliases = {
                "高等数学": ("高等数学", "微积分", "calculus"),
                "大学物理": ("大学物理", "物理", "physics"),
                "程序设计": ("程序设计", "算法设计", "计算机程序设计", "programming", "algorithm"),
            }
            return next(
                (family for family, names in aliases.items() if normalized in names),
                normalized,
            )

        expected = canonical(goal.course)
        declared_courses = [*skill.courses, *([skill.source_course] if skill.source_course else [])]
        return bool(declared_courses) and any(canonical(course) == expected for course in declared_courses)

    @staticmethod
    def _topic_matches(skill: TeachingSkill, goal: TeachingGoal) -> bool:
        if skill.skill_type != "subject":
            return True
        # The topic title is the strongest intent anchor. Knowledge points and
        # generic words in the objective are often shared across adjacent
        # chapters (for example “合力” appears in both Newton's first- and
        # second-law teaching), so one incidental objective hit must not admit
        # a neighboring Skill. If the title is broad, require at least two
        # declared target tags in the objective before admitting the Skill.
        compact_topic = re.sub(r"\s+", "", goal.topic.lower())
        compact_objective = re.sub(r"\s+", "", goal.objective.lower())
        goal_context = compact_topic + compact_objective
        required_tags = [
            re.sub(r"\s+", "", tag.lower())
            for tag in skill.required_topic_tags
            if tag.strip()
        ]
        if any(tag not in goal_context for tag in required_tags):
            return False
        tags = [re.sub(r"\s+", "", tag.lower()) for tag in skill.topic_tags if tag.strip()]
        if not tags:
            return False
        title_hits = [tag for tag in tags if tag in compact_topic]
        if title_hits:
            return True
        objective_hits = [tag for tag in tags if tag in compact_objective]
        return len(objective_hits) >= min(2, len(tags))

    @staticmethod
    def _trigger_matches(skill: TeachingSkill, state: StudentState, student_message: str) -> bool:
        if skill.skill_type != "subject":
            return True
        if not student_message.strip():
            stage = "initial"
        elif state.misconceptions:
            stage = "misconception"
        elif any(
            cue in student_message
            for cue in ("不知道", "不懂", "不会", "不明白", "混淆", "记不住", "搞不清")
        ):
            stage = "confusion"
        else:
            stage = "followup"
        return stage in skill.trigger_states

    @staticmethod
    def _preconditions_match(
        skill: TeachingSkill,
        goal: TeachingGoal,
        state: StudentState,
        profile: StudentProfile | None,
        student_message: str,
    ) -> bool:
        low, high = skill.mastery_range
        average = state.average_mastery()
        if not low <= average <= high:
            return False
        if skill.skill_type == "subject" and skill.required_prior_knowledge:
            if average >= skill.prerequisite_mastery_bypass:
                return True
            observed = re.sub(
                r"\s+",
                "",
                " ".join([*(profile.prior_knowledge if profile else []), student_message]).lower(),
            )
            return any(
                re.sub(r"\s+", "", item.lower()) in observed
                for item in skill.required_prior_knowledge
                if item.strip()
            )
        return True

    def candidates(
        self,
        goal: TeachingGoal,
        state: StudentState,
        student_message: str = "",
        *,
        profile: StudentProfile | None = None,
        limit: int = 5,
        include_generic: bool = False,
    ) -> tuple[list[TeachingSkill], list[dict[str, Any]]]:
        """Hard filter, then deterministically rank, returning a full audit trail."""
        audit: list[dict[str, Any]] = []
        eligible: list[TeachingSkill] = []
        for skill in self.skills:
            reasons: list[str] = []
            type_ok = include_generic or skill.skill_type == "subject"
            course_ok = self._course_matches(skill, goal) if skill.skill_type == "subject" else None
            topic_ok = self._topic_matches(skill, goal) if skill.skill_type == "subject" else None
            trigger_ok = (
                self._trigger_matches(skill, state, student_message) if skill.skill_type == "subject" else None
            )
            precondition_ok = (
                self._preconditions_match(skill, goal, state, profile, student_message)
                if skill.skill_type == "subject"
                else None
            )
            if not type_ok:
                reasons.append("非学科候选")
            if course_ok is False:
                reasons.append("课程不匹配")
            if topic_ok is False:
                reasons.append("教学目标不匹配")
            if trigger_ok is False:
                reasons.append("当前触发阶段不匹配")
            if precondition_ok is False:
                reasons.append("前置条件或掌握区间不满足")
            passed = type_ok and all(check is not False for check in (course_ok, topic_ok, trigger_ok, precondition_ok))
            if passed:
                reasons.append("课程、目标、触发阶段与前置条件通过")
                eligible.append(skill)
            audit.append(
                {
                    "skill_id": skill.skill_id,
                    "passed": passed,
                    "reasons": reasons,
                    "score": 0.0,
                    "checks": {
                        "type": type_ok,
                        "course": course_ok,
                        "goal": topic_ok,
                        "trigger": trigger_ok,
                        "precondition": precondition_ok,
                    },
                }
            )

        if not eligible:
            return [], audit
        ranked_all = self.rank(goal, state, student_message, limit=len(self.skills), include_generic=True)
        ranked = [skill for skill in ranked_all if skill in eligible][:limit]
        rank_score = {skill.skill_id: float(len(ranked) - index) for index, skill in enumerate(ranked)}
        for item in audit:
            item["score"] = rank_score.get(item["skill_id"], 0.0)
        return ranked, audit
