import requests
import streamlit as st
from datetime import datetime
import random

BACKEND_URL = "http://127.0.0.1:8000/api/chat"

st.set_page_config(
    page_title="가상 환자",
    page_icon="🩺",
    layout="wide",
)

# --------------------------------------------------
# Case definitions
# --------------------------------------------------
CASE_LIBRARY = {
    "case_1_appendicitis": {
        "case_title": "Appendicitis",
        "thumbnail": "🟥",
        "name": "김철수",
        "age": 23,
        "sex": "남성",
        "chief_complaint": "복통",
        "department": "응급의학과",
        "difficulty": "쉬움",
        "associated_symptoms": ["메스꺼움", "식욕부진"],
    },
    "case_2_cholecystitis": {
        "case_title": "Cholecystitis",
        "thumbnail": "🟨",
        "name": "이영희",
        "age": 46,
        "sex": "여성",
        "chief_complaint": "우상복부 통증",
        "department": "소화기내과",
        "difficulty": "중간",
        "associated_symptoms": ["오심", "발열"],
    },
    "case_3_pyelonephritis": {
        "case_title": "Pyelonephritis",
        "thumbnail": "🟦",
        "name": "박민지",
        "age": 31,
        "sex": "여성",
        "chief_complaint": "옆구리 통증",
        "department": "신장내과",
        "difficulty": "중간",
        "associated_symptoms": ["발열", "배뇨통"],
    },
}

CASE_OPTIONS = {
    "Appendicitis": "case_1_appendicitis",
    "Cholecystitis": "case_2_cholecystitis",
    "Pyelonephritis": "case_3_pyelonephritis",
}

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "start"

if "selected_case_id" not in st.session_state:
    st.session_state.selected_case_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "encounter_notes" not in st.session_state:
    st.session_state.encounter_notes = ""


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def make_initial_messages() -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "안녕하세요 선생님. 무엇이 불편해서 오셨나요?",
            "timestamp": now_str(),
        }
    ]


def start_case(case_id: str) -> None:
    st.session_state.selected_case_id = case_id
    st.session_state.messages = make_initial_messages()
    st.session_state.encounter_notes = ""
    st.session_state.page = "chat"


def reset_conversation() -> None:
    st.session_state.messages = make_initial_messages()
    st.session_state.encounter_notes = ""


def go_to_start() -> None:
    st.session_state.page = "start"
    st.session_state.selected_case_id = None
    st.session_state.messages = []
    st.session_state.encounter_notes = ""


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "timestamp": now_str(),
        }
    )


def get_current_case() -> dict:
    case_id = st.session_state.selected_case_id
    return CASE_LIBRARY[case_id]


def call_backend(user_input: str) -> str:
    payload = {
        "case_id": st.session_state.selected_case_id,
        "message": user_input,
        "history": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1]
        ],
    }

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["reply"]
    except Exception as e:
        return f"Backend error: {e}"


def handle_user_message(user_input: str) -> None:
    add_message("user", user_input)
    reply = call_backend(user_input)
    add_message("assistant", reply)


# --------------------------------------------------
# Start page
# --------------------------------------------------
def render_start_page() -> None:
    st.title("🩺 가상 환자 시뮬레이터")
    st.caption("케이스 카드를 보고 원하는 환자를 선택하세요.")

    top1, top2 = st.columns([1, 1])

    with top1:
        if st.button("🎲 랜덤 케이스 시작", use_container_width=True):
            random_case_id = random.choice(list(CASE_LIBRARY.keys()))
            start_case(random_case_id)
            st.rerun()

    with top2:
        st.info("각 케이스를 선택하면 바로 문진을 시작할 수 있습니다.")

    st.markdown("---")
    st.subheader("케이스 선택")

    case_items = list(CASE_LIBRARY.items())

    for row_start in range(0, len(case_items), 2):
        cols = st.columns(2)
        row_cases = case_items[row_start:row_start + 2]

        for col, (case_id, case_data) in zip(cols, row_cases):
            with col:
                with st.container(border=True):
                    st.markdown(f"## {case_data['thumbnail']} {case_data['case_title']}")
                    st.write(
                        f"**환자:** {case_data['name']} ({case_data['age']}세, {case_data['sex']})"
                    )
                    st.write(f"**주호소:** {case_data['chief_complaint']}")
                    st.write(f"**진료과:** {case_data['department']}")
                    st.write(f"**난이도:** {case_data['difficulty']}")
                    st.write(
                        f"**힌트:** {', '.join(case_data['associated_symptoms'][:2])}"
                    )

                    if st.button(
                        "이 케이스 시작",
                        key=f"start_{case_id}",
                        use_container_width=True,
                    ):
                        start_case(case_id)
                        st.rerun()


# --------------------------------------------------
# Chat page
# --------------------------------------------------
def render_chat_page() -> None:
    patient = get_current_case()

    with st.sidebar:
        st.title("🩺 가상 환자")
        st.caption("대화 중심 UI")

        st.subheader("현재 케이스")
        st.write(f"**케이스명:** {patient['case_title']}")
        st.write(f"**이름:** {patient['name']}")
        st.write(f"**나이/성별:** {patient['age']} / {patient['sex']}")
        st.write(f"**주호소:** {patient['chief_complaint']}")

        st.divider()

        st.subheader("빠른 기능")
        if st.button("대화 초기화", use_container_width=True):
            reset_conversation()
            st.rerun()

        if st.button("케이스 다시 선택", use_container_width=True):
            go_to_start()
            st.rerun()

        if st.button("정답 보기", use_container_width=True):
            st.json(
                {
                    "case_id": st.session_state.selected_case_id,
                    **patient,
                }
            )

        st.divider()

        st.subheader("진료 메모")
        st.session_state.encounter_notes = st.text_area(
            "요약 작성",
            value=st.session_state.encounter_notes,
            height=220,
            placeholder="예:\n- RLQ 복통\n- 6시간 전 시작\n- 메스꺼움 있음\n- 수술력 없음",
        )

    st.title("가상 환자 면담")
    st.caption("자연스럽게 병력청취를 진행하세요.")

    left_col, right_col = st.columns([3, 1])

    with left_col:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                st.caption(message["timestamp"])

        prompt = st.chat_input("환자에게 질문하세요...")

        if prompt:
            handle_user_message(prompt)
            st.rerun()

    with right_col:
        st.subheader("면담 팁")
        st.markdown(
            """
- 개방형 질문으로 시작
- OPQRST 순서로 진행
- 동반/부정 증상 확인
- 마지막에 요약
"""
        )

        st.subheader("추천 질문")
        suggestions = [
            "어디가 불편해서 오셨어요?",
            "언제부터 아프셨나요?",
            "어디가 아프신가요?",
            "통증은 어떤 느낌인가요?",
            "동반 증상이 있나요?",
            "수술이나 병력 있으신가요?",
        ]

        for i, suggestion in enumerate(suggestions):
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                handle_user_message(suggestion)
                st.rerun()

    with st.expander("현재 백엔드 연결 정보"):
        st.code(
            f"""POST {BACKEND_URL}

payload = {{
    "case_id": "<selected_case_id>",
    "message": "<user_input>",
    "history": [...]
}}""",
            language="python",
        )


# --------------------------------------------------
# Router
# --------------------------------------------------
if st.session_state.page == "start":
    render_start_page()
else:
    render_chat_page()