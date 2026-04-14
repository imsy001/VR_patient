import requests
import streamlit as st
from datetime import datetime
import random

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="가상 환자",
    page_icon="🩺",
    layout="wide",
)

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "start"

if "selected_case_id" not in st.session_state:
    st.session_state.selected_case_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "encounter_notes" not in st.session_state:
    st.session_state.encounter_notes = ""

if "case_list" not in st.session_state:
    st.session_state.case_list = []

if "current_case" not in st.session_state:
    st.session_state.current_case = None


# --------------------------------------------------
# Backend API
# --------------------------------------------------
def fetch_case_list():
    try:
        res = requests.get(f"{BACKEND_URL}/api/cases")
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"케이스 목록 불러오기 실패: {e}")
        return []


def fetch_case_detail(case_id: str):
    try:
        res = requests.get(f"{BACKEND_URL}/api/case/{case_id}")
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"케이스 로딩 실패: {e}")
        return None


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
        response = requests.post(f"{BACKEND_URL}/api/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["reply"]
    except Exception as e:
        return f"Backend error: {e}"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def now_str():
    return datetime.now().strftime("%H:%M:%S")


def make_initial_messages():
    return [{
        "role": "assistant",
        "content": "안녕하세요 선생님.",
        "timestamp": now_str(),
    }]


def start_case(case_id: str):
    case_data = fetch_case_detail(case_id)

    if not case_data:
        return

    st.session_state.selected_case_id = case_id
    st.session_state.current_case = case_data
    st.session_state.messages = make_initial_messages()
    st.session_state.page = "chat"


def reset_conversation():
    st.session_state.messages = make_initial_messages()


def go_to_start():
    st.session_state.page = "start"
    st.session_state.selected_case_id = None
    st.session_state.current_case = None
    st.session_state.messages = []


def add_message(role: str, content: str):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": now_str(),
    })


def handle_user_message(user_input: str):
    add_message("user", user_input)
    reply = call_backend(user_input)
    add_message("assistant", reply)


# --------------------------------------------------
# Start page
# --------------------------------------------------
def render_start_page():
    st.title("🩺 가상 환자 시뮬레이터")

    if not st.session_state.case_list:
        st.session_state.case_list = fetch_case_list()

    case_list = st.session_state.case_list

    if not case_list:
        st.warning("케이스 없음")
        return

    if st.button("🎲 랜덤 케이스 시작"):
        case_id = random.choice([c["case_id"] for c in case_list])
        start_case(case_id)
        st.rerun()

    st.markdown("---")

    for case in case_list:
        with st.container(border=True):
            st.subheader(case["case_title"])
            st.write(f"ID: {case['case_id']}")

            if st.button("시작", key=case["case_id"]):
                start_case(case["case_id"])
                st.rerun()


# --------------------------------------------------
# Chat page
# --------------------------------------------------
def render_chat_page():
    case = st.session_state.current_case

    patient = case.get("patient_info", {})

    with st.sidebar:
        st.subheader("현재 케이스")

        st.write(f"케이스: {case.get('case_title')}")
        st.write(f"이름: {patient.get('name')}")
        st.write(f"나이/성별: {patient.get('age')} / {patient.get('sex')}")
        st.write(f"주호소: {case.get('chief_complaint')}")

        if st.button("초기화"):
            reset_conversation()
            st.rerun()

        if st.button("뒤로"):
            go_to_start()
            st.rerun()

        if st.button("정답 보기"):
            st.json(case)

    st.title("가상 환자 면담")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            st.caption(msg["timestamp"])

    prompt = st.chat_input("질문하세요")

    if prompt:
        handle_user_message(prompt)
        st.rerun()


# --------------------------------------------------
# Router
# --------------------------------------------------
if st.session_state.page == "start":
    render_start_page()
else:
    render_chat_page()