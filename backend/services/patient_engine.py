import random

from backend.services.llm_service import (
    classify_question_intent,
    naturalize_patient_answer,
)

NON_FACT_INTENTS = {
    "greeting",
    "thanks",
    "goodbye",
    "small_talk",
    "unknown",
    "specific_symptom",
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


def _choose(options: list[str]) -> str:
    return random.choice(options)


def _safe_get_dict(data: dict, key: str) -> dict:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _safe_get_list(data: dict, key: str) -> list[str]:
    value = data.get(key, [])
    return value if isinstance(value, list) else []


def _get_patient_info(patient_context: dict) -> dict:
    return _safe_get_dict(patient_context, "patient_info")


def _get_vital_signs(patient_context: dict) -> dict:
    return _safe_get_dict(patient_context, "vital_signs")


def _get_history_taking(patient_context: dict) -> dict:
    return _safe_get_dict(patient_context, "history_taking")


def _get_past_history(patient_context: dict) -> dict:
    return _safe_get_dict(patient_context, "past_history")


def _flatten_dict_to_korean_lines(data: dict) -> list[str]:
    items: list[str] = []

    for key, value in data.items():
        if isinstance(value, dict):
            nested_parts = []
            for sub_key, sub_value in value.items():
                if sub_value:
                    nested_parts.append(f"{sub_key}: {sub_value}")
            if nested_parts:
                items.append(f"{key}({', '.join(nested_parts)})")
        else:
            if value:
                items.append(f"{key}: {value}")

    return items


def _flatten_past_history(patient_context: dict) -> list[str]:
    past_history = _get_past_history(patient_context)
    if not past_history:
        return []

    items: list[str] = []

    trauma = past_history.get("trauma")
    if trauma:
        items.append(f"외상력은 {trauma}")

    hospitalization = past_history.get("hospitalization")
    if hospitalization:
        items.append(f"입원력은 {hospitalization}")

    medication = past_history.get("medication")
    if medication:
        items.append(f"복용 약은 {medication}")

    family_history = past_history.get("family_history")
    if family_history:
        items.append(f"가족력은 {family_history}")

    medical_history = _safe_get_dict(past_history, "medical_history")
    if medical_history:
        med_bits = []
        for key in ["HTN", "DM", "TB", "hepatitis"]:
            value = medical_history.get(key)
            if value:
                med_bits.append(f"{key}는 {value}")
        if med_bits:
            items.append(", ".join(med_bits))

    social_history = _safe_get_dict(past_history, "social_history")
    if social_history:
        social_map = {
            "alcohol": "음주",
            "smoking": "흡연",
            "coffee": "커피",
            "occupation": "직업",
        }
        social_bits = []
        for key, label in social_map.items():
            value = social_history.get(key)
            if value:
                social_bits.append(f"{label}는 {value}")
        if social_bits:
            items.append(", ".join(social_bits))

    gynecologic_history = _safe_get_dict(past_history, "gynecologic_history")
    if gynecologic_history:
        gyn_map = {
            "LMP": "마지막 생리는",
            "menstrual_cycle": "생리 주기는",
            "pregnancy_possibility": "임신 가능성은",
        }
        gyn_bits = []
        for key, label in gyn_map.items():
            value = gynecologic_history.get(key)
            if value:
                gyn_bits.append(f"{label} {value}")
        if gyn_bits:
            items.append(", ".join(gyn_bits))

    return items


def _normalize_text(value) -> str:
    return str(value).strip().lower()


def _match_canonical_symptom(user_message: str) -> str | None:
    msg = _normalize_text(user_message)

    for canonical, aliases in SYMPTOM_ALIASES.items():
        if any(alias.lower() in msg for alias in aliases):
            return canonical

    return None


def _build_symptom_fact_map(patient_context: dict) -> dict[str, str]:
    result: dict[str, str] = {}

    positives = patient_context.get("history_taking", {}).get("associated_symptoms", []) or []
    negatives = patient_context.get("denied_symptoms", []) or []

    for symptom in positives:
        result[str(symptom).strip()] = "present"

    for symptom in negatives:
        result[str(symptom).strip()] = "absent"

    return result


def _yes_no_symptom_style_response(canonical_symptom: str, status: str) -> str:
    positive_map = {
        "발열": ["네, 열이 나요.", "네, 열이 있어요."],
        "오한": ["네, 오한도 있어요.", "네, 춥고 오한이 있어요."],
        "배뇨 시 불편감": [
            "네, 소변 볼 때 불편해요.",
            "네, 소변 볼 때 좀 아프고 불편해요.",
        ],
        "빈뇨": [
            "네, 소변이 자주 마려워요.",
            "네, 화장실을 자주 가게 돼요.",
        ],
        "흉통": ["아니요, 가슴 통증은 없어요."],
        "설사": ["아니요, 설사는 없어요."],
        "두통": [
            "머리 아픈 건 잘 모르겠어요.",
            "잘 모르겠어요. 머리 아픈 건 잘 모르겠어요.",
        ],
        "구토": ["잘 모르겠어요."],
        "메스꺼움": ["잘 모르겠어요."],
        "어지럼": ["잘 모르겠어요."],
        "기침": ["잘 모르겠어요."],
        "호흡곤란": ["잘 모르겠어요."],
        "복통": ["배도 조금 불편한 느낌은 있어요."],
        "옆구리 통증": ["네, 오른쪽 옆구리가 아파요."],
    }

    negative_map = {
        "발열": ["아니요, 열은 없어요."],
        "오한": ["아니요, 오한은 없어요."],
        "배뇨 시 불편감": ["아니요, 소변 볼 때 불편한 건 없어요."],
        "빈뇨": ["아니요, 소변이 자주 마렵진 않아요."],
        "흉통": ["아니요, 흉통은 없어요.", "아니요, 가슴은 안 아파요."],
        "설사": ["아니요, 설사는 없어요."],
        "두통": ["아니요, 머리 아픈 건 없어요."],
        "구토": ["아니요, 토한 적은 없어요."],
        "메스꺼움": ["아니요, 메스꺼운 건 없어요."],
        "어지럼": ["아니요, 어지럽진 않아요."],
        "기침": ["아니요, 기침은 없어요."],
        "호흡곤란": ["아니요, 숨찬 건 없어요."],
        "복통": ["아니요, 배 아픈 건 없어요."],
        "옆구리 통증": ["아니요, 옆구리 통증은 없어요."],
    }

    if status == "present":
        return _choose(positive_map.get(canonical_symptom, ["네, 있어요."]))

    return _choose(negative_map.get(canonical_symptom, ["아니요, 없어요."]))


def _detect_specific_symptom_answer(
    user_message: str,
    patient_context: dict,
) -> str | None:
    canonical = _match_canonical_symptom(user_message)
    if canonical is None:
        return None

    symptom_fact_map = _build_symptom_fact_map(patient_context)
    status = symptom_fact_map.get(canonical)

    if status is None:
        history_taking = _get_history_taking(patient_context)
        chief_complaint = patient_context.get("chief_complaint", "")

        if canonical == "옆구리 통증":
            location = history_taking.get("location")
            if location and "옆구리" in str(location):
                status = "present"

        elif canonical == "복통":
            referred = history_taking.get("referred_pain", "")
            combined = f"{referred} {chief_complaint}"
            if "배" in combined:
                status = "present"

        elif canonical == "두통":
            return "머리 아픈 건 잘 모르겠어요."

    if status is None:
        return None

    return _yes_no_symptom_style_response(canonical, status)


def _lookup_fact(intent: str, user_message: str, patient_context: dict):
    patient_info = _get_patient_info(patient_context)
    vital_signs = _get_vital_signs(patient_context)
    history_taking = _get_history_taking(patient_context)
    past_history = _get_past_history(patient_context)

    if intent == "greeting":
        return "__greeting__"

    if intent == "thanks":
        return "__thanks__"

    if intent == "goodbye":
        return "__goodbye__"

    if intent == "small_talk":
        return "__small_talk__"

    if intent == "name":
        return patient_info.get("name")

    if intent == "age":
        return patient_info.get("age")

    if intent == "sex":
        sex = patient_info.get("sex")
        if sex == "female":
            return "여자"
        if sex == "male":
            return "남자"
        return sex

    if intent == "chief_complaint":
        return patient_context.get("chief_complaint")

    if intent == "onset":
        return history_taking.get("onset")

    if intent == "duration":
        return history_taking.get("duration")

    if intent == "course":
        return history_taking.get("course")

    if intent == "location":
        return history_taking.get("location")

    if intent == "character":
        return history_taking.get("character")

    if intent == "severity":
        return history_taking.get("severity")

    if intent == "migration":
        return history_taking.get("migration")

    if intent == "referred_pain":
        return history_taking.get("referred_pain")

    if intent == "associated_symptoms":
        symptoms = _safe_get_list(history_taking, "associated_symptoms")
        return symptoms if symptoms else None

    if intent == "aggravating_factors":
        factors = _safe_get_list(history_taking, "aggravating_factors")
        return factors if factors else None

    if intent == "relieving_factors":
        factors = _safe_get_list(history_taking, "relieving_factors")
        return factors if factors else None

    if intent == "denied_symptoms":
        denied = _safe_get_list(patient_context, "denied_symptoms")
        return denied if denied else None

    if intent == "similar_episode":
        return patient_context.get("similar_episode")

    if intent == "previous_examination":
        return patient_context.get("previous_examination")

    if intent == "past_history":
        history_items = _flatten_past_history(patient_context)
        return history_items if history_items else None

    if intent == "medication":
        return past_history.get("medication")

    if intent == "family_history":
        return past_history.get("family_history")

    if intent == "social_history":
        social_history = _safe_get_dict(past_history, "social_history")
        if not social_history:
            return None

        social_map = {
            "alcohol": "음주",
            "smoking": "흡연",
            "coffee": "커피",
            "occupation": "직업",
        }
        result = []
        for key, value in social_history.items():
            if value:
                label = social_map.get(key, key)
                result.append(f"{label}는 {value}")
        return result if result else None

    if intent == "gynecologic_history":
        gynecologic_history = _safe_get_dict(past_history, "gynecologic_history")
        if not gynecologic_history:
            return None

        gyn_map = {
            "LMP": "마지막 생리는",
            "menstrual_cycle": "생리 주기는",
            "pregnancy_possibility": "임신 가능성은",
        }
        result = []
        for key, value in gynecologic_history.items():
            if value:
                label = gyn_map.get(key, key)
                result.append(f"{label} {value}")
        return result if result else None

    if intent == "vital_signs":
        if not vital_signs:
            return None

        result = []
        if vital_signs.get("blood_pressure"):
            result.append(f"혈압은 {vital_signs.get('blood_pressure')}")
        if vital_signs.get("pulse_rate"):
            result.append(f"맥박은 {vital_signs.get('pulse_rate')}")
        if vital_signs.get("respiratory_rate"):
            result.append(f"호흡수는 {vital_signs.get('respiratory_rate')}")
        if vital_signs.get("temperature"):
            result.append(f"체온은 {vital_signs.get('temperature')}")
        return result if result else None

    return None


def _lookup_facts(
    primary_intent: str,
    secondary_intent: str | None,
    user_message: str,
    patient_context: dict,
) -> dict:
    raw_values = {}

    if primary_intent != "unknown":
        primary_value = _lookup_fact(primary_intent, user_message, patient_context)
        if primary_value is not None:
            raw_values[primary_intent] = primary_value

    if secondary_intent and secondary_intent not in {"unknown", primary_intent}:
        secondary_value = _lookup_fact(secondary_intent, user_message, patient_context)
        if secondary_value is not None:
            raw_values[secondary_intent] = secondary_value

    return raw_values


def _generate_small_talk_reply(user_message: str) -> str:
    msg = _normalize_text(user_message)

    if "안녕" in msg:
        return "안녕하세요."

    if "감사" in msg or "고마" in msg:
        return "아니에요."

    if "오시는데" in msg or "오는 데" in msg or "불편한 점은 없" in msg:
        return _choose([
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
        return _choose([
            "네, 알겠습니다.",
            "네.",
        ])

    if "괜찮" in msg or "가능" in msg or "해도 될까요" in msg or "되실까요" in msg:
        return _choose([
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
        return _choose([
            "네.",
            "네, 알겠습니다.",
            "네, 괜찮아요.",
        ])

    if "불편하시면" in msg or "불편하시다면" in msg or "아프면" in msg or "말씀해주세요" in msg:
        return _choose([
            "네, 알겠습니다.",
            "네, 말씀드릴게요.",
            "네.",
        ])

    if "다음" in msg or "몇 가지만" in msg or "계속" in msg or "이제" in msg:
        return _choose([
            "네.",
            "네, 알겠습니다.",
        ])

    return "네."


def _python_fallback_format(intent: str, raw_value, user_message: str | None = None) -> str:
    if intent == "greeting":
        return _choose([
            "안녕하세요.",
            "안녕하세요, 선생님.",
        ])

    if intent == "thanks":
        return _choose([
            "아니에요.",
            "아니에요, 감사합니다.",
        ])

    if intent == "goodbye":
        return _choose([
            "네, 감사합니다.",
            "감사합니다. 안녕히 계세요.",
        ])

    if intent == "small_talk":
        return _generate_small_talk_reply(user_message or "")

    if raw_value is None:
        return _choose([
            "잘 모르겠어요. 다시 말씀해 주시겠어요?",
            "죄송한데 잘 못 알아들었어요. 한 번만 다시 말씀해 주세요.",
        ])

    if intent == "name":
        return _choose([
            f"{raw_value}입니다.",
            f"제 이름은 {raw_value}입니다.",
        ])

    if intent == "age":
        return _choose([
            f"{raw_value}살이에요.",
            f"저는 {raw_value}살입니다.",
        ])

    if intent == "sex":
        return _choose([
            f"{raw_value}예요.",
            f"{raw_value}입니다.",
        ])

    if intent == "chief_complaint":
        return _choose([
            f"{raw_value} 때문에 왔어요.",
            f"{raw_value}가 있어서 왔어요.",
        ])

    if intent == "onset":
        return str(raw_value)

    if intent == "duration":
        return str(raw_value)

    if intent == "course":
        return str(raw_value)

    if intent == "location":
        return _choose([
            f"{raw_value}.",
            f"{raw_value}가 아파요.",
        ])

    if intent == "character":
        return _choose([
            f"{raw_value}.",
            f"{raw_value}한 느낌이에요.",
        ])

    if intent == "severity":
        return _choose([
            f"{raw_value}.",
            f"통증은 {raw_value} 정도예요.",
        ])

    if intent == "migration":
        return str(raw_value)

    if intent == "referred_pain":
        return str(raw_value)

    if intent == "medication":
        return str(raw_value)

    if intent == "family_history":
        return str(raw_value)

    if intent == "similar_episode":
        return str(raw_value)

    if isinstance(raw_value, str):
        return raw_value

    if isinstance(raw_value, list):
        joined = ", ".join(str(v) for v in raw_value)

        if intent == "associated_symptoms":
            return _choose([
                f"{joined}도 있어요.",
                f"{joined} 같은 증상도 있었어요.",
            ])

        if intent == "denied_symptoms":
            return _choose([
                f"{joined}는 없었어요.",
                f"{joined} 같은 증상은 없어요.",
            ])

        if intent in {
            "past_history",
            "social_history",
            "gynecologic_history",
            "vital_signs",
            "aggravating_factors",
            "relieving_factors",
        }:
            return joined

        return joined

    if isinstance(raw_value, dict):
        lines = _flatten_dict_to_korean_lines(raw_value)
        return ", ".join(lines) if lines else "잘 모르겠어요."

    return str(raw_value)


def _python_fallback_format_multi(
    primary_intent: str,
    secondary_intent: str | None,
    raw_values: dict,
    user_message: str,
) -> str:
    if primary_intent == "greeting":
        return _python_fallback_format("greeting", None, user_message)

    if primary_intent == "thanks":
        return _python_fallback_format("thanks", None, user_message)

    if primary_intent == "goodbye":
        return _python_fallback_format("goodbye", None, user_message)

    if primary_intent == "small_talk":
        return _python_fallback_format("small_talk", None, user_message)

    if not raw_values:
        return _choose([
            "잘 모르겠어요. 다시 말씀해 주시겠어요?",
            "죄송한데 잘 못 알아들었어요. 다시 말씀해 주세요.",
        ])

    primary_text = _python_fallback_format(
        primary_intent,
        raw_values.get(primary_intent),
        user_message,
    )

    if secondary_intent and secondary_intent in raw_values:
        secondary_text = _python_fallback_format(
            secondary_intent,
            raw_values.get(secondary_intent),
            user_message,
        )
        if primary_text == secondary_text:
            return primary_text
        return f"{primary_text} {secondary_text}"

    return primary_text


def generate_reply(
    user_message: str,
    patient_context: dict,
    history: list[dict],
) -> tuple[str, str]:
    """
    Architecture:
    1) direct yes/no symptom question is handled first
    2) GPT classifies primary + optional secondary intent
    3) backend deterministically selects the facts
    4) GPT naturalizes the facts into Korean patient speech
    5) Python fallback is used only as a last resort
    """

    direct_symptom_answer = _detect_specific_symptom_answer(user_message, patient_context)
    if direct_symptom_answer is not None:
        print(f"[DIRECT SYMPTOM] {user_message} -> {direct_symptom_answer}")
        return direct_symptom_answer, "specific_symptom"

    intent, secondary_intent = classify_question_intent(user_message, history)
    print(f"[INTENT] {user_message} -> primary={intent}, secondary={secondary_intent}")

    if intent == "unknown":
        return _choose([
            "잘 모르겠어요. 다시 말씀해 주시겠어요?",
            "죄송한데 잘 못 알아들었어요. 다시 한번 말씀해 주세요.",
        ]), "unknown"

    if intent == "small_talk":
        reply = _generate_small_talk_reply(user_message)
        print(f"[FINAL REPLY] {reply}")
        return reply, "small_talk"

    raw_values = _lookup_facts(intent, secondary_intent, user_message, patient_context)
    print(f"[RAW FACTS] {raw_values}")

    try:
        reply = naturalize_patient_answer(
            intent=intent,
            secondary_intent=secondary_intent,
            raw_values=raw_values,
            user_message=user_message,
            history=history,
        )
    except Exception as e:
        print(f"[NATURALIZE ERROR] {e}")
        reply = _python_fallback_format_multi(intent, secondary_intent, raw_values, user_message)

    if not reply.strip():
        reply = _python_fallback_format_multi(intent, secondary_intent, raw_values, user_message)

    combined_intent = f"{intent}+{secondary_intent}" if secondary_intent else intent
    print(f"[FINAL REPLY] {reply}")

    return reply, combined_intent