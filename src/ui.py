from __future__ import annotations

import streamlit as st

from src.agent import HybridTeachingAgent
from src.config import get_agent_settings
from src.llm import OpenAICompatibleClient
from src.models import StudentState, TeachingSession
from src.skills import SkillLibrary
from src.storage import EvaluationStore, SessionStore

ACTION_LABELS = {
    "subject_instruction": "学科讲解",
    "diagnostic": "诊断提问",
    "scaffold": "分层提示",
    "correction": "误解纠正",
    "transfer": "迁移验证",
    "terminate_success": "成功终止",
    "terminate_unable": "暂停教学",
    "terminate_max_rounds": "达到轮数上限",
    "terminate_no_improvement": "纠错后仍未改善",
    "fixed_verification": "固定策略验证",
    "generic": "通用讲解",
    "generic_verification": "通用验证",
}

DECISION_MODE_LABELS = {
    "llm_semantic_selection": "LLM 语义选择",
    "rule_fallback": "规则回退",
    "rule_margin_selection": "规则分差选择",
    "deterministic_guard": "确定性守卫",
    "termination_guard": "终止守卫",
    "candidate_constraint_fallback": "候选约束回退",
    "security_guard": "安全守卫",
}

STATUS_LABELS = {"active": "教学中", "success": "已达成", "unable": "已暂停"}


@st.cache_resource(max_entries=1)
def get_library() -> SkillLibrary:
    return SkillLibrary()


@st.cache_resource(max_entries=1)
def get_llm_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient()


@st.cache_resource(max_entries=1)
def get_store() -> SessionStore:
    return SessionStore()


@st.cache_resource(max_entries=1)
def get_evaluation_store() -> EvaluationStore:
    return EvaluationStore()


def build_agent(*, fast_demo: bool = False) -> HybridTeachingAgent:
    settings = get_agent_settings()
    if fast_demo:
        # Live teaching remains a real-LLM path, but does not pay the full
        # evaluation review budget on every ordinary turn. Deterministic
        # single-step checks still run; semantic review is only added when a
        # draft actually fails those checks.
        settings.update(
            {
                "fast_demo_mode": True,
                "max_state_reviews": 1,
                "state_review_call_budget": 1,
                "semantic_selector_margin": 0.0,
                "structured_output_retries": 0,
            }
        )
    return HybridTeachingAgent(
        library=get_library(),
        llm=get_llm_client(),
        store=get_store(),
        settings=settings,
    )


def initialize_state() -> None:
    st.session_state.setdefault("teaching_session", None)
    st.session_state.setdefault("evaluation_report", None)
    st.session_state.setdefault("selected_replay", None)
    st.session_state.setdefault("selected_evaluation", None)
    st.session_state.setdefault("student_reply", None)
    st.session_state.setdefault("ui_notice", None)
    # Streamlit hot reload can keep an instance created by the previous model
    # class in browser memory. Revalidate it so newly added optional fields get
    # their defaults instead of raising AttributeError after a schema update.
    session = st.session_state.get("teaching_session")
    if session is not None and (
        # A Streamlit rerun can retain an object whose class came from the
        # previous module instance.  `hasattr` alone is not enough here: the
        # object may expose every new field while still failing Pydantic's
        # nested-model validation when a new ConversationTurn is appended.
        not isinstance(session, TeachingSession)
        or not isinstance(getattr(session, "state", None), StudentState)
        or not hasattr(session, "teaching_route")
        or not hasattr(session, "rounds_in_current_run")
    ):
        st.session_state.teaching_session = TeachingSession.model_validate(session.model_dump(mode="python"))


def page_header(title: str, caption: str, *, eyebrow: str, icon: str) -> None:
    """Render a consistent, restrained page identity using native elements."""
    with st.container(gap=None):
        st.caption(f":material/{icon}:　{eyebrow}")
        st.title(title)
        st.caption(caption)


def inject_live_chat_css() -> None:
    """Give the live page a calm ChatGPT-like conversation canvas.

    The selectors intentionally target Streamlit's public ``data-testid``
    attributes instead of private generated class names, so the visual layer
    remains stable across reruns and minor Streamlit upgrades.
    """
    st.markdown(
        """
        <style>
        /* Conversation canvas */
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(201, 216, 212, .78);
            border-radius: 18px 18px 18px 6px;
            padding: .65rem .9rem;
            margin-top: .7rem !important;
            margin-bottom: .7rem !important;
            width: fit-content !important;
            max-width: min(76%, 820px) !important;
            background: rgba(255, 255, 255, .9);
            box-shadow: 0 5px 18px rgba(18, 54, 56, .035);
        }
        /* Streamlit 1.59 stores the role on stChatMessageContent.  These
           selectors deliberately outrank Streamlit's centered chat layout. */
        [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageContent"][aria-label="Chat message from assistant"]
        ) {
            margin-left: 0 !important;
            margin-right: auto !important;
            border-color: #c9d8d4 !important;
            background: rgba(255, 255, 255, .94) !important;
        }
        [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageContent"][aria-label="Chat message from user"]
        ) {
            margin-left: auto !important;
            margin-right: 0 !important;
            max-width: min(68%, 740px) !important;
            border-radius: 18px 18px 6px 18px !important;
            border-color: #9fc9be !important;
            background: #dff1eb !important;
            box-shadow: 0 5px 18px rgba(15, 118, 110, .08) !important;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            line-height: 1.78;
            font-size: 1rem;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {
            margin-bottom: .15rem;
        }
        /* Keep the input visually attached to the conversation */
        [data-testid="stChatInput"] {
            max-width: 920px;
            margin: .75rem auto 1.5rem;
        }
        [data-testid="stChatInput"] textarea {
            border-radius: 18px !important;
            min-height: 52px !important;
            padding: .8rem 3.2rem .8rem 1rem !important;
            background: #f1f7f5 !important;
            border: 1px solid #b8d0ca !important;
            box-shadow: 0 7px 24px rgba(18, 54, 56, .07) !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: #0f766e !important;
            box-shadow: 0 0 0 3px rgba(15, 118, 110, .13) !important;
        }
        /* Current task card: less dashboard, more lesson */
        [data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
            margin-top: .75rem;
        }
        @media (max-width: 720px) {
            [data-testid="stChatMessage"] {
                border-radius: 14px;
                padding: .5rem .7rem;
                margin: .4rem 0;
                max-width: 88% !important;
            }
            [data-testid="stChatMessage"]:has(
                [data-testid="stChatMessageContent"][aria-label="Chat message from user"]
            ) {
                max-width: 84% !important;
                border-radius: 14px 14px 5px 14px !important;
            }
            [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
                font-size: .95rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def action_label(value: str) -> str:
    return ACTION_LABELS.get(value, value)


def decision_mode_label(value: str) -> str:
    return DECISION_MODE_LABELS.get(value, value)


def set_notice(message: str, icon: str = ":material/check_circle:") -> None:
    st.session_state.ui_notice = (message, icon)


def show_notice() -> None:
    notice = st.session_state.pop("ui_notice", None)
    if notice:
        st.toast(notice[0], icon=notice[1])


def reset_teaching_session() -> None:
    st.session_state.teaching_session = None
    st.session_state.student_reply = None
    set_notice("已保存当前记录，可以创建新的教学会话。", ":material/restart_alt:")
