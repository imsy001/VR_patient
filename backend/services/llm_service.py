import json
from typing import Literal, Optional

from openai import OpenAI
from backend.config import OPENAI_API_KEY, OPENAI_MODEL

IntentLabel = Literal[
    "greeting",
    "thanks",
    "goodbye",
    "small_talk",
    "name",
    "age",
    "sex",
    "chief_complaint",
    "onset",
    "location",
    "character",
    "severity",
    "associated_symptoms",
    "denied_symptoms",
    "past_history",
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
    "location",
    "character",
    "severity",
    "associated_symptoms",
    "denied_symptoms",
    "past_history",
    "unknown",
}

NON_FACT_INTENTS = {"greeting", "thanks", "goodbye", "small_talk", "unknown"}

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _serialize_raw_value(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return ", ".join(str(x) for x in raw_value)
    return str(raw_value)


def _history_to_text(history: list[dict], limit: int = 6) -> str:
    return "\n".join(
        f"{item.get('role', 'unknown')}: {item.get('content', '')}"
        for item in history[-limit:]
    )


def _generate_small_talk_response(user_message: str) -> str:
    msg = user_message.strip().lower()

    # Permission / cooperation
    if "괜찮" in msg or "가능" in msg or "해도 될까요" in msg or "되실까요" in msg:
        return "네, 괜찮아요."

    # Physical exam / procedural transition
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
        return "네."

    # Reassurance / instruction-related
    if "불편" in msg or "아프면" in msg or "말씀" in msg:
        return "네, 알겠습니다."

    # Transition language
    if "다음" in msg or "몇 가지만" in msg or "계속" in msg or "이제" in msg:
        return "네."

    return "네."


def classify_question_intent(
    user_message: str,
    history: list[dict],
) -> tuple[IntentLabel, Optional[IntentLabel]]:
    if client is None:
        return "unknown", None

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
location
character
severity
associated_symptoms
denied_symptoms
past_history
unknown

Definitions:
- greeting: opening greeting to the patient
- thanks: thanking the patient
- goodbye: closing the encounter
- small_talk: non-medical conversational or procedural utterances that do not ask for a patient fact,
  including permission, transitions, exam instructions, reassurance, or conversational acknowledgments
- associated_symptoms: asking what other symptoms are present
- denied_symptoms: asking what symptoms are absent / denied

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

Examples:
"안녕하세요" -> {"intent":"greeting","secondary_intent":null}
"감사합니다" -> {"intent":"thanks","secondary_intent":null}
"안녕히 계세요" -> {"intent":"goodbye","secondary_intent":null}
"성함이 어떻게 되세요?" -> {"intent":"name","secondary_intent":null}
"몇 살이세요?" -> {"intent":"age","secondary_intent":null}
"남자세요 여자세요?" -> {"intent":"sex","secondary_intent":null}
"어디가 제일 불편하세요?" -> {"intent":"chief_complaint","secondary_intent":null}
"언제부터 아프셨어요?" -> {"intent":"onset","secondary_intent":null}
"어디가 어떻게 아프세요?" -> {"intent":"location","secondary_intent":"character"}
"언제부터 얼마나 아프셨어요?" -> {"intent":"onset","secondary_intent":"severity"}
"다른 증상도 있었나요?" -> {"intent":"associated_symptoms","secondary_intent":null}
"열이나 설사는 없었나요?" -> {"intent":"denied_symptoms","secondary_intent":null}
"예전 수술이나 큰 병은 있으셨어요?" -> {"intent":"past_history","secondary_intent":null}
"문진이랑 신체진찰 진행하겠습니다. 괜찮으실까요?" -> {"intent":"small_talk","secondary_intent":null}
"이제 배 좀 만져볼게요." -> {"intent":"small_talk","secondary_intent":null}
"좋습니다. 몇 가지만 더 여쭤볼게요." -> {"intent":"small_talk","secondary_intent":null}
"불편하시면 말씀해주세요." -> {"intent":"small_talk","secondary_intent":null}
"잠깐만요." -> {"intent":"small_talk","secondary_intent":null}
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

        # Small talk should not carry a secondary medical intent
        if primary == "small_talk":
            secondary = None

        return primary, secondary

    except Exception as e:
        print(f"[LLM CLASSIFY ERROR] {e}")
        return "unknown", None


def naturalize_patient_answer(
    intent: str,
    secondary_intent: Optional[str],
    raw_values: dict,
    user_message: str,
    patient_context: dict,
    history: list[dict],
) -> str:
    if client is None:
        if intent == "greeting":
            return "안녕하세요."
        if intent == "thanks":
            return "네."
        if intent == "goodbye":
            return "감사합니다."
        if intent == "small_talk":
            return _generate_small_talk_response(user_message)
        if raw_values:
            parts = []
            for _, value in raw_values.items():
                s = _serialize_raw_value(value)
                if s:
                    parts.append(s)
            return " ".join(parts) if parts else "잘 모르겠어요."
        return "잘 모르겠어요."

    # Non-fact conversational intents
    if intent == "greeting":
        return "안녕하세요."
    if intent == "thanks":
        return "아니에요."
    if intent == "goodbye":
        return "네, 감사합니다."
    if intent == "small_talk":
        return _generate_small_talk_response(user_message)

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
- Do NOT contradict the provided slot values or patient context.
- Do NOT explain medically.
- Speak only as the patient.

Style rules:
- Use natural spoken Korean.
- Usually 1-2 sentences.
- If both primary and secondary intents are present, combine them naturally in one response.
- You may use light hesitation naturally, such as:
  "음...", "잘은 모르겠는데", "한...", "좀", "약간"
- Rephrase the slot values into everyday patient language.
- Avoid robotic, list-like, or textbook-like phrasing.

Special rules:
- Mention only the provided slot values.
- If one of the two requested slot values is missing, answer naturally using only the available one.
- If both are missing, say naturally that you are not sure.

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
        "patient_context": {
            "name": patient_context.get("name"),
            "sex": patient_context.get("sex"),
            "age": patient_context.get("age"),
        },
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
        return "잘 모르겠어요."


def generate_reply(
    user_message: str,
    patient_context: dict,
    history: list[dict],
) -> tuple[str, str]:
    intent, secondary_intent = classify_question_intent(user_message, history)

    raw_values = {}

    if intent not in NON_FACT_INTENTS:
        raw_values[intent] = patient_context.get(intent)

    if secondary_intent and secondary_intent not in NON_FACT_INTENTS:
        raw_values[secondary_intent] = patient_context.get(secondary_intent)

    reply = naturalize_patient_answer(
        intent=intent,
        secondary_intent=secondary_intent,
        raw_values=raw_values,
        user_message=user_message,
        patient_context=patient_context,
        history=history,
    )

    combined_intent = f"{intent}+{secondary_intent}" if secondary_intent else intent
    return reply, combined_intent