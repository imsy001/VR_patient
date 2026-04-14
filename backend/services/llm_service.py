import json
import random
from typing import Literal, Optional

from openai import OpenAI
from backend.config import OPENAI_API_KEY, OPENAI_MODEL


IntentLabel = Literal[
    # social / conversational
    "greeting",
    "thanks",
    "goodbye",
    "small_talk",

    # demographics
    "name",
    "age",
    "sex",

    # chief complaint
    "chief_complaint",

    # HPI
    "onset",
    "duration",
    "course",
    "location",
    "character",
    "severity",
    "migration",
    "referred_pain",
    "associated_symptoms",
    "aggravating_factors",
    "relieving_factors",

    # negatives / PMH
    "denied_symptoms",
    "past_history",
    "medication",
    "family_history",
    "social_history",
    "gynecologic_history",
    "similar_episode",

    # direct symptom Q/A
    "specific_symptom",

    "unknown",
]

ALLOWED_INTENTS = {
    "greeting",
    "thanks",
    "goodbye",
    "small_talk",
    "name",
    "age",
    "sex",
    "chief_complaint",
    "onset",
    "duration",
    "course",
    "location",
    "character",
    "severity",
    "migration",
    "referred_pain",
    "associated_symptoms",
    "aggravating_factors",
    "relieving_factors",
    "denied_symptoms",
    "past_history",
    "medication",
    "family_history",
    "social_history",
    "gynecologic_history",
    "similar_episode",
    "specific_symptom",
    "unknown",
}

NON_FACT_INTENTS = {
    "greeting",
    "thanks",
    "goodbye",
    "small_talk",
    "unknown",
}

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


INTENT_TO_FIELD_PATH: dict[str, tuple[str, ...]] = {
    "name": ("patient_info", "name"),
    "age": ("patient_info", "age"),
    "sex": ("patient_info", "sex"),
    "chief_complaint": ("chief_complaint",),
    "onset": ("history_taking", "onset"),
    "duration": ("history_taking", "duration"),
    "course": ("history_taking", "course"),
    "location": ("history_taking", "location"),
    "character": ("history_taking", "character"),
    "severity": ("history_taking", "severity"),
    "migration": ("history_taking", "migration"),
    "referred_pain": ("history_taking", "referred_pain"),
    "associated_symptoms": ("history_taking", "associated_symptoms"),
    "aggravating_factors": ("history_taking", "aggravating_factors"),
    "relieving_factors": ("history_taking", "relieving_factors"),
    "denied_symptoms": ("denied_symptoms",),
    "past_history": ("past_history",),
    "medication": ("past_history", "medication"),
    "family_history": ("past_history", "family_history"),
    "social_history": ("past_history", "social_history"),
    "gynecologic_history": ("past_history", "gynecologic_history"),
    "similar_episode": ("similar_episode",),
}


SYMPTOM_ALIASES: dict[str, list[str]] = {
    "발열": ["발열", "열", "fever"],
    "오한": ["오한", "춥", "몸살", "chill", "chills"],
    "배뇨 시 불편감": [
        "배뇨 시 불편감",
        "소변 볼 때 불편",
        "소변 볼 때 아프",
        "소변 볼 때 따갑",
        "배뇨통",
        "dysuria",
    ],
    "빈뇨": [
        "빈뇨",
        "소변이 자주",
        "자주 마려",
        "화장실을 자주",
        "frequency",
        "urinary frequency",
    ],
    "흉통": ["흉통", "가슴 통증", "가슴 아프", "chest pain"],
    "설사": ["설사", "diarrhea", "diarrhoea"],
    "두통": ["두통", "머리 아프", "headache", "head hurts"],
    "구토": ["구토", "토", "vomit", "vomiting"],
    "메스꺼움": ["메스껍", "오심", "nausea"],
    "어지럼": ["어지럽", "어지럼", "dizziness", "dizzy"],
    "기침": ["기침", "cough"],
    "호흡곤란": ["숨차", "호흡곤란", "숨이 차", "shortness of breath", "sob"],
    "복통": ["복통", "배 아프", "abd pain", "abdominal pain"],
    "옆구리 통증": ["옆구리 아프", "옆구리 통증", "flank pain"],
}


def _get_nested_value(data: dict, path: tuple[str, ...]):
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _serialize_raw_value(raw_value):
    if raw_value is None:
        return None

    if isinstance(raw_value, list):
        return ", ".join(str(x) for x in raw_value)

    if isinstance(raw_value, dict):
        parts = []
        for key, value in raw_value.items():
            if isinstance(value, dict):
                nested_parts = []
                for sub_key, sub_value in value.items():
                    nested_parts.append(f"{sub_key}: {sub_value}")
                parts.append(f"{key}({', '.join(nested_parts)})")
            else:
                parts.append(f"{key}: {value}")
        return "; ".join(parts)

    return str(raw_value)


def _history_to_text(history: list[dict], limit: int = 6) -> str:
    return "\n".join(
        f"{item.get('role', 'unknown')}: {item.get('content', '')}"
        for item in history[-limit:]
    )


def _normalize_text(value) -> str:
    return str(value).strip().lower()


def _build_symptom_fact_map(patient_context: dict) -> dict[str, str]:
    """
    Returns canonical symptom -> status
    status in {"present", "absent"}
    """
    result: dict[str, str] = {}

    positives = patient_context.get("history_taking", {}).get("associated_symptoms", []) or []
    negatives = patient_context.get("denied_symptoms", []) or []

    for symptom in positives:
        s = str(symptom).strip()
        result[s] = "present"

    for symptom in negatives:
        s = str(symptom).strip()
        result[s] = "absent"

    return result


def _match_canonical_symptom(user_message: str) -> Optional[str]:
    msg = _normalize_text(user_message)

    for canonical, aliases in SYMPTOM_ALIASES.items():
        if any(alias.lower() in msg for alias in aliases):
            return canonical

    return None


def _yes_no_symptom_style_response(canonical_symptom: str, status: str) -> str:
    positive_map = {
        "발열": [
            "네, 열이 나요.",
            "네, 열이 있어요.",
        ],
        "오한": [
            "네, 오한도 있어요.",
            "네, 춥고 오한이 있어요.",
        ],
        "배뇨 시 불편감": [
            "네, 소변 볼 때 불편해요.",
            "네, 소변 볼 때 좀 아프고 불편해요.",
        ],
        "빈뇨": [
            "네, 소변이 자주 마려워요.",
            "네, 화장실을 자주 가게 돼요.",
        ],
        "흉통": [
            "네, 가슴 통증이 있어요.",
        ],
        "설사": [
            "네, 설사가 있어요.",
        ],
        "두통": [
            "네, 머리도 아파요.",
        ],
        "구토": [
            "네, 토한 적 있어요.",
        ],
        "메스꺼움": [
            "네, 메스꺼워요.",
        ],
        "어지럼": [
            "네, 좀 어지러워요.",
        ],
        "기침": [
            "네, 기침이 있어요.",
        ],
        "호흡곤란": [
            "네, 좀 숨이 차요.",
        ],
        "복통": [
            "네, 배도 아파요.",
        ],
        "옆구리 통증": [
            "네, 오른쪽 옆구리가 아파요.",
        ],
    }

    negative_map = {
        "발열": [
            "아니요, 열은 없어요.",
        ],
        "오한": [
            "아니요, 오한은 없어요.",
        ],
        "배뇨 시 불편감": [
            "아니요, 소변 볼 때 불편한 건 없어요.",
        ],
        "빈뇨": [
            "아니요, 소변이 자주 마렵진 않아요.",
        ],
        "흉통": [
            "아니요, 흉통은 없어요.",
            "아니요, 가슴은 안 아파요.",
        ],
        "설사": [
            "아니요, 설사는 없어요.",
        ],
        "두통": [
            "아니요, 머리 아픈 건 없어요.",
        ],
        "구토": [
            "아니요, 토한 적은 없어요.",
        ],
        "메스꺼움": [
            "아니요, 메스꺼운 건 없어요.",
        ],
        "어지럼": [
            "아니요, 어지럽진 않아요.",
        ],
        "기침": [
            "아니요, 기침은 없어요.",
        ],
        "호흡곤란": [
            "아니요, 숨찬 건 없어요.",
        ],
        "복통": [
            "아니요, 배 아픈 건 없어요.",
        ],
        "옆구리 통증": [
            "아니요, 옆구리 통증은 없어요.",
        ],
    }

    if status == "present":
        choices = positive_map.get(canonical_symptom, ["네, 있어요."])
        return random.choice(choices)

    choices = negative_map.get(canonical_symptom, ["아니요, 없어요."])
    return random.choice(choices)


def _detect_specific_symptom_answer(
    user_message: str,
    patient_context: dict,
) -> Optional[str]:
    """
    Handles direct symptom questions like:
    - 머리 아프신가요?
    - 열이 나세요?
    - 설사는 없으세요?
    """
    canonical = _match_canonical_symptom(user_message)
    if canonical is None:
        return None

    symptom_fact_map = _build_symptom_fact_map(patient_context)
    status = symptom_fact_map.get(canonical)

    # Special handling for symptoms that are explicitly described outside lists
    if status is None:
        if canonical == "옆구리 통증":
            location = _get_nested_value(patient_context, ("history_taking", "location"))
            if location and "옆구리" in str(location):
                status = "present"

        elif canonical == "복통":
            referred = _get_nested_value(patient_context, ("history_taking", "referred_pain"))
            chief = patient_context.get("chief_complaint")
            combined = f"{referred or ''} {chief or ''}"
            if "배" in combined:
                status = "present"

        elif canonical == "두통":
            return "머리 아픈 건 잘 모르겠어요."

    if status is None:
        return None

    return _yes_no_symptom_style_response(canonical, status)


def _generate_small_talk_response(user_message: str) -> str:
    """
    small_talk is intentionally rule-based.
    It must NOT reveal new medical facts.
    """
    msg = user_message.strip().lower()

    if "안녕" in msg:
        return "안녕하세요."

    if "감사" in msg or "고마" in msg:
        return "아니에요."

    if "오시는데" in msg or "오는 데" in msg or "불편한 점은 없" in msg:
        return random.choice([
            "아니요, 괜찮았어요.",
            "오는 데는 괜찮았어요.",
            "특별히 불편한 건 없었어요.",
        ])

    if (
        "여성력" in msg
        or "산과력" in msg
        or "월경력" in msg
        or "생리력" in msg
        or "질문하겠" in msg
        or "여쭤보겠" in msg
        or "확인하겠" in msg
    ):
        return random.choice([
            "네, 알겠습니다.",
            "네.",
        ])

    if "괜찮" in msg or "가능" in msg or "해도 될까요" in msg or "되실까요" in msg:
        return random.choice([
            "네, 괜찮아요.",
            "네, 괜찮습니다.",
            "네.",
        ])

    if (
        "진찰" in msg
        or "신체진찰" in msg
        or "문진" in msg
        or "만져" in msg
        or "눌러" in msg
        or "볼게요" in msg
        or "보겠습니다" in msg
        or "검사" in msg
    ):
        return random.choice([
            "네.",
            "네, 알겠습니다.",
            "네, 괜찮아요.",
        ])

    if "불편하시면" in msg or "불편하시다면" in msg or "아프면" in msg or "말씀해주세요" in msg:
        return random.choice([
            "네, 알겠습니다.",
            "네, 말씀드릴게요.",
            "네.",
        ])

    if "다음" in msg or "몇 가지만" in msg or "계속" in msg or "이제" in msg:
        return random.choice([
            "네.",
            "네, 알겠습니다.",
        ])

    return "네."


def _fallback_keyword_intent(user_message: str) -> tuple[IntentLabel, Optional[IntentLabel]]:
    msg = user_message.strip().lower()

    # social
    if "안녕" in msg:
        return "greeting", None
    if "감사" in msg or "고마" in msg:
        return "thanks", None
    if "안녕히" in msg or "들어가세요" in msg:
        return "goodbye", None

    # direct symptom Q/A first
    if _match_canonical_symptom(msg) is not None:
        return "specific_symptom", None

    # procedural / small talk
    if (
        "괜찮" in msg
        or "가능" in msg
        or "해도 될까요" in msg
        or "되실까요" in msg
        or "문진" in msg
        or "신체진찰" in msg
        or "진찰" in msg
        or "검사" in msg
        or "만져" in msg
        or "눌러" in msg
        or "볼게요" in msg
        or "보겠습니다" in msg
        or "불편하시면" in msg
        or "불편하시다면" in msg
        or "말씀해주세요" in msg
        or "오시는데" in msg
        or "오는 데" in msg
        or "질문하겠" in msg
        or "여쭤보겠" in msg
        or "확인하겠" in msg
        or "여성력" in msg
        or "산과력" in msg
        or "월경력" in msg
        or "생리력" in msg
    ):
        return "small_talk", None

    # demographics
    if "성함" in msg or "이름" in msg:
        return "name", None
    if "몇 살" in msg or "나이" in msg:
        return "age", None
    if "남자" in msg or "여자" in msg or "성별" in msg:
        return "sex", None

    # chief complaint
    if "어디가 제일 불편" in msg or "무엇 때문에" in msg or "왜 오셨" in msg:
        return "chief_complaint", None

    # HPI
    if "언제부터" in msg or "몇 일 전" in msg or "며칠 전" in msg or "시작" in msg:
        return "onset", None
    if "얼마나 오래" in msg or "지속" in msg:
        return "duration", None
    if "점점" in msg or "심해졌" in msg or "나아졌" in msg:
        return "course", None
    if "어디가 아프" in msg or "위치" in msg:
        return "location", None
    if "어떻게 아프" in msg or "느낌" in msg or "양상" in msg:
        return "character", None
    if "몇 점" in msg or "통증 정도" in msg:
        return "severity", None
    if "옮겨" in msg or "퍼졌" in msg or "번졌" in msg:
        return "migration", None
    if "다른 곳도 아프" in msg or "방사" in msg:
        return "referred_pain", None
    if "더 심해" in msg or "악화" in msg:
        return "aggravating_factors", None
    if "좀 나아" in msg or "완화" in msg or "덜 아프" in msg:
        return "relieving_factors", None

    # symptoms
    if "다른 증상" in msg or "같이" in msg or "동반" in msg:
        return "associated_symptoms", None
    if "없었" in msg or "없나요" in msg or "부인" in msg:
        return "denied_symptoms", None

    # history
    if "과거력" in msg or "예전 수술" in msg or "큰 병" in msg:
        return "past_history", None
    if "약 드시" in msg or "복용" in msg:
        return "medication", None
    if "가족력" in msg:
        return "family_history", None
    if "술" in msg or "담배" in msg or "직업" in msg or "커피" in msg:
        return "social_history", None
    if "생리" in msg or "마지막 월경" in msg or "임신 가능성" in msg:
        return "gynecologic_history", None
    if "비슷하게 아픈 적" in msg or "예전에 이런 적" in msg:
        return "similar_episode", None

    return "unknown", None


def classify_question_intent(
    user_message: str,
    history: list[dict],
) -> tuple[IntentLabel, Optional[IntentLabel]]:
    if client is None:
        return _fallback_keyword_intent(user_message)

    history_text = _history_to_text(history, limit=4)

    instructions = """
You are an intent classifier for a virtual patient medical interview.

Classify the doctor's latest utterance into:
- one primary intent: "intent"
- optionally one secondary intent: "secondary_intent"

Allowed labels:
greeting
thanks
goodbye
small_talk
name
age
sex
chief_complaint
onset
duration
course
location
character
severity
migration
referred_pain
associated_symptoms
aggravating_factors
relieving_factors
denied_symptoms
past_history
medication
family_history
social_history
gynecologic_history
similar_episode
specific_symptom
unknown

Definitions:
- greeting: opening greeting to the patient
- thanks: thanking the patient
- goodbye: closing the encounter
- small_talk: non-medical conversational or procedural utterances that do not ask for a patient fact,
  including permission, transitions, exam instructions, reassurance, conversational acknowledgments,
  and simple logistical comfort checks
- onset: when the symptom started
- duration: how long it has continued
- course: how it changed over time
- migration: whether pain moved from one area to another
- referred_pain: whether discomfort is felt in another related area
- associated_symptoms: symptoms present together with the main complaint
- aggravating_factors: what makes it worse
- relieving_factors: what makes it better
- denied_symptoms: symptoms absent / denied
- past_history: broad past medical/surgical history when not more specific
- medication: current medication use
- family_history: family history
- social_history: alcohol, smoking, coffee, occupation, etc.
- gynecologic_history: LMP, menstrual cycle, pregnancy possibility, etc.
- similar_episode: prior similar episodes
- specific_symptom: direct yes/no question about a particular symptom
  such as headache, fever, chills, dysuria, diarrhea, chest pain, etc.

Rules:
- Return ONLY valid JSON.
- Return a single-line JSON object only.
- No markdown.
- No code fences.
- Output format must be exactly:
{"intent":"<one_label>","secondary_intent":"<one_label_or_null>"}
- "intent" is required.
- Include "secondary_intent": null when absent.
- Use the single best primary intent.
- Use secondary_intent only when the utterance clearly asks for two medical facts.
- Do not return more than one secondary intent.
- Do not duplicate the same label in both fields.
- If unclear or not mappable, return unknown with secondary_intent null.
- For direct yes/no symptom questions, prefer specific_symptom over broader categories.
- Statements like "질문하겠습니다", "여쭤보겠습니다", "확인하겠습니다" are usually small_talk
  unless the doctor is actually asking the medical question now.

Examples:
"안녕하세요" -> {"intent":"greeting","secondary_intent":null}
"감사합니다" -> {"intent":"thanks","secondary_intent":null}
"안녕히 계세요" -> {"intent":"goodbye","secondary_intent":null}
"성함이 어떻게 되세요?" -> {"intent":"name","secondary_intent":null}
"몇 살이세요?" -> {"intent":"age","secondary_intent":null}
"남자세요 여자세요?" -> {"intent":"sex","secondary_intent":null}
"어디가 제일 불편하세요?" -> {"intent":"chief_complaint","secondary_intent":null}
"언제부터 아프셨어요?" -> {"intent":"onset","secondary_intent":null}
"얼마나 계속 아프셨어요?" -> {"intent":"duration","secondary_intent":null}
"시간 지나면서 더 심해졌나요?" -> {"intent":"course","secondary_intent":null}
"어디가 어떻게 아프세요?" -> {"intent":"location","secondary_intent":"character"}
"통증이 다른 데로 퍼지나요?" -> {"intent":"migration","secondary_intent":null}
"아랫배 쪽으로도 불편하세요?" -> {"intent":"referred_pain","secondary_intent":null}
"무엇을 하면 더 아프세요?" -> {"intent":"aggravating_factors","secondary_intent":null}
"가만히 있으면 좀 나아지나요?" -> {"intent":"relieving_factors","secondary_intent":null}
"다른 증상도 있었나요?" -> {"intent":"associated_symptoms","secondary_intent":null}
"열이나 설사는 없었나요?" -> {"intent":"denied_symptoms","secondary_intent":null}
"머리 아프세요?" -> {"intent":"specific_symptom","secondary_intent":null}
"열이 나세요?" -> {"intent":"specific_symptom","secondary_intent":null}
"예전 수술이나 큰 병은 있으셨어요?" -> {"intent":"past_history","secondary_intent":null}
"현재 드시는 약 있으세요?" -> {"intent":"medication","secondary_intent":null}
"가족 중에 비슷한 병 앓으신 분 있나요?" -> {"intent":"family_history","secondary_intent":null}
"술, 담배는 어떻게 하세요?" -> {"intent":"social_history","secondary_intent":null}
"마지막 생리는 언제였나요?" -> {"intent":"gynecologic_history","secondary_intent":null}
"예전에 비슷하게 아팠던 적 있나요?" -> {"intent":"similar_episode","secondary_intent":null}
"문진이랑 신체진찰 진행하겠습니다. 괜찮으실까요?" -> {"intent":"small_talk","secondary_intent":null}
"이제 배 좀 만져볼게요." -> {"intent":"small_talk","secondary_intent":null}
"오시는데 불편한 점은 없으셨나요?" -> {"intent":"small_talk","secondary_intent":null}
"불편하시다면 말씀해주세요." -> {"intent":"small_talk","secondary_intent":null}
"여성력 관련해서 질문하겠습니다." -> {"intent":"small_talk","secondary_intent":null}
"배아픈 것 외에 다른 증상 있는지 여쭤보겠습니다." -> {"intent":"small_talk","secondary_intent":null}
""".strip()

    input_text = f"""
Conversation history:
{history_text}

Doctor's latest utterance:
{user_message}
""".strip()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=input_text,
        )

        text = response.output_text.strip()
        data = json.loads(text)

        primary = data.get("intent", "unknown")
        secondary = data.get("secondary_intent", None)

        if primary not in ALLOWED_INTENTS:
            primary = "unknown"

        if secondary not in ALLOWED_INTENTS:
            secondary = None

        if secondary == primary:
            secondary = None

        if primary in {"small_talk", "specific_symptom"}:
            secondary = None

        return primary, secondary

    except Exception as e:
        print(f"[LLM CLASSIFY ERROR] {e}")
        return _fallback_keyword_intent(user_message)


def _naturalize_small_talk_with_llm(
    user_message: str,
    history: list[dict],
) -> str:
    """
    This function is left for fallback/experimentation,
    but generate_reply() will not use LLM for small_talk.
    """
    if client is None:
        return _generate_small_talk_response(user_message)

    history_text = _history_to_text(history, limit=6)

    instructions = """
You are roleplaying as a patient in a Korean medical interview.

The doctor's latest utterance is small talk, procedural speech, reassurance, or a simple logistical check.

CRITICAL RULE:
- If the doctor is NOT asking for medical information,
  you MUST ONLY acknowledge briefly.
- NEVER provide any symptom, history, or medical detail.
- EVEN IF you know the patient's condition, DO NOT mention it.

Allowed responses:
- 네.
- 네, 알겠습니다.
- 네, 괜찮아요.
- 네, 말씀드릴게요.
- 아니요, 괜찮았어요.

Strict constraints:
- Maximum 1 sentence
- Do NOT add explanation
- Do NOT expand
- Do NOT continue conversation

Output:
- Return only the patient's utterance.
- No markdown.
- No quotation marks.
""".strip()

    payload = {
        "doctor_utterance": user_message,
        "recent_conversation": history_text,
    }

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
        )
        text = response.output_text.strip()
        if text:
            return text
    except Exception as e:
        print(f"[LLM SMALL TALK ERROR] {e}")

    return _generate_small_talk_response(user_message)


def naturalize_patient_answer(
    intent: str,
    secondary_intent: Optional[str],
    raw_values: dict,
    user_message: str,
    history: list[dict],
) -> str:
    if intent == "greeting":
        return "안녕하세요."
    if intent == "thanks":
        return "아니에요."
    if intent == "goodbye":
        return "네, 감사합니다."

    if intent == "small_talk":
        return _generate_small_talk_response(user_message)

    if client is None:
        if raw_values:
            parts = []
            for _, value in raw_values.items():
                s = _serialize_raw_value(value)
                if s:
                    parts.append(s)
            return " ".join(parts) if parts else "잘 모르겠어요."
        return "잘 모르겠어요."

    serialized_raw_values = {}
    for key, value in raw_values.items():
        serialized = _serialize_raw_value(value)
        if serialized is not None:
            serialized_raw_values[key] = serialized

    if not serialized_raw_values:
        return "잘 모르겠어요."

    history_text = _history_to_text(history, limit=6)

    instructions = """
You are roleplaying as a patient in a Korean medical interview.

Your goal:
- Answer like a real patient in natural spoken Korean.
- Preserve the facts exactly from the provided slot values.
- If the doctor asked about two things, answer both naturally in one response.

Hard constraints:
- Do NOT invent new medical facts.
- Do NOT add new symptoms, diagnoses, timelines, locations, severity values, or history.
- Do NOT mention any information that is not explicitly contained in slot_values.
- Even if you know other facts about the patient, do NOT reveal them.
- Do NOT explain medically.
- Speak only as the patient.

Style rules:
- Use natural spoken Korean.
- Usually 1 short sentence, sometimes 2 short sentences.
- If both primary and secondary intents are present, combine them naturally in one response.
- You may use light hesitation naturally, such as:
  "음...", "잘은 모르겠는데", "한...", "좀", "약간"
- Rephrase the slot values into everyday patient language.
- Avoid robotic, list-like, or textbook-like phrasing.

Special rules:
- Answer only the exact information requested.
- Mention only the provided slot values.
- Do not add other symptoms or history.
- If one of the two requested slot values is missing, answer naturally using only the available one.
- If both are missing, say naturally that you are not sure.
- For structured histories like social_history, gynecologic_history, or past_history,
  summarize only the facts directly present in slot_values.

Output:
- Return only the patient's utterance.
- No markdown.
- No quotation marks.
""".strip()

    payload = {
        "primary_intent": intent,
        "secondary_intent": secondary_intent,
        "slot_values": serialized_raw_values,
        "doctor_question": user_message,
        "recent_conversation": history_text,
    }

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
        )
        text = response.output_text.strip()
        return text if text else "잘 모르겠어요."
    except Exception as e:
        print(f"[LLM NATURALIZE ERROR] {e}")

        parts = []
        for _, value in raw_values.items():
            s = _serialize_raw_value(value)
            if s:
                parts.append(s)
        return " ".join(parts) if parts else "잘 모르겠어요."


def _extract_raw_value(patient_context: dict, intent: str):
    path = INTENT_TO_FIELD_PATH.get(intent)
    if path is None:
        return None
    return _get_nested_value(patient_context, path)


def generate_reply(
    user_message: str,
    patient_context: dict,
    history: list[dict],
) -> tuple[str, str]:
    # 1. Direct symptom yes/no questions first
    direct_symptom_answer = _detect_specific_symptom_answer(user_message, patient_context)
    if direct_symptom_answer is not None:
        return direct_symptom_answer, "specific_symptom"

    # 2. Intent classification
    intent, secondary_intent = classify_question_intent(user_message, history)

    # 3. small_talk must NEVER reveal medical information
    if intent == "small_talk":
        return _generate_small_talk_response(user_message), "small_talk"

    raw_values = {}

    if intent not in NON_FACT_INTENTS and intent != "specific_symptom":
        raw_values[intent] = _extract_raw_value(patient_context, intent)

    if secondary_intent and secondary_intent not in NON_FACT_INTENTS:
        raw_values[secondary_intent] = _extract_raw_value(patient_context, secondary_intent)

    reply = naturalize_patient_answer(
        intent=intent,
        secondary_intent=secondary_intent,
        raw_values=raw_values,
        user_message=user_message,
        history=history,
    )

    combined_intent = f"{intent}+{secondary_intent}" if secondary_intent else intent
    return reply, combined_intent