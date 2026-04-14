import random

from backend.services.llm_service import (
    classify_question_intent,
    naturalize_patient_answer,
)


NON_FACT_INTENTS = {"greeting", "thanks", "goodbye", "small_talk", "unknown"}


def _choose(options: list[str]) -> str:
    return random.choice(options)


def _normalize_text_list(values: list[str]) -> set[str]:
    return {str(v).strip().lower() for v in values}


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


def _flatten_past_history(patient_context: dict) -> list[str]:
    """
    Convert structured past_history into readable Korean snippets
    for fallback use.
    """
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
    medical_bits = []
    for key in ["HTN", "DM", "TB", "hepatitis"]:
        value = medical_history.get(key)
        if value:
            medical_bits.append(f"{key} {value}")
    if medical_bits:
        items.append("과거력은 " + ", ".join(medical_bits))

    social_history = _safe_get_dict(past_history, "social_history")
    social_bits = []
    social_map = {
        "alcohol": "음주",
        "smoking": "흡연",
        "coffee": "커피",
        "occupation": "직업",
    }
    for key, label in social_map.items():
        value = social_history.get(key)
        if value:
            social_bits.append(f"{label}는 {value}")
    if social_bits:
        items.append(", ".join(social_bits))

    gynecologic_history = _safe_get_dict(past_history, "gynecologic_history")
    gyn_bits = []
    gyn_map = {
        "LMP": "마지막 생리는",
        "menstrual_cycle": "생리 주기는",
        "pregnancy_possibility": "임신 가능성은",
    }
    for key, label in gyn_map.items():
        value = gynecologic_history.get(key)
        if value:
            gyn_bits.append(f"{label} {value}")
    if gyn_bits:
        items.append(", ".join(gyn_bits))

    return items


def _match_explicit_denied_symptoms(user_message: str, patient_context: dict) -> list[str]:
    """
    Return explicitly matched denied symptoms only.
    """
    lower_msg = user_message.lower()
    denied = _normalize_text_list(_safe_get_list(patient_context, "denied_symptoms"))

    symptom_aliases = {
        "설사": ["설사", "diarrhea"],
        "흉통": ["흉통", "chest pain", "chestpain"],
        "구토": ["구토", "vomiting", "vomit"],
        "발열": ["발열", "fever", "열"],
        "오심": ["오심", "nausea"],
        "식욕부진": ["식욕부진", "loss of appetite", "appetite loss"],
    }

    matched = []

    for canonical, aliases in symptom_aliases.items():
        if any(alias in lower_msg for alias in aliases):
            if canonical.lower() in denied or any(alias in denied for alias in aliases):
                matched.append(canonical)

    return matched


def _lookup_fact(intent: str, user_message: str, patient_context: dict):
    patient_info = _get_patient_info(patient_context)
    history_taking = _get_history_taking(patient_context)

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

    if intent == "location":
        return history_taking.get("location")

    if intent == "duration":
        return history_taking.get("duration")

    if intent == "course":
        return history_taking.get("course")

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
        explicit_denied = _match_explicit_denied_symptoms(user_message, patient_context)
        if explicit_denied:
            return explicit_denied

        denied = _safe_get_list(patient_context, "denied_symptoms")
        return denied if denied else None

    if intent == "similar_episode":
        return patient_context.get("similar_episode")

    if intent == "previous_examination":
        return patient_context.get("previous_examination")

    if intent == "past_history":
        history_items = _flatten_past_history(patient_context)
        return history_items if history_items else None

    if intent == "medical_history":
        medical_history = _safe_get_dict(_get_past_history(patient_context), "medical_history")
        if not medical_history:
            return None
        return [f"{k} {v}" for k, v in medical_history.items() if v]

    if intent == "social_history":
        social_history = _safe_get_dict(_get_past_history(patient_context), "social_history")
        if not social_history:
            return None
        social_map = {
            "alcohol": "음주",
            "smoking": "흡연",
            "coffee": "커피",
            "occupation": "직업",
        }
        return [f"{social_map.get(k, k)}는 {v}" for k, v in social_history.items() if v]

    if intent == "gynecologic_history":
        gynecologic_history = _safe_get_dict(_get_past_history(patient_context), "gynecologic_history")
        if not gynecologic_history:
            return None
        gyn_map = {
            "LMP": "마지막 생리는",
            "menstrual_cycle": "생리 주기는",
            "pregnancy_possibility": "임신 가능성은",
        }
        return [f"{gyn_map.get(k, k)} {v}" for k, v in gynecologic_history.items() if v]

    if intent == "vital_signs":
        vital_signs = _get_vital_signs(patient_context)
        if not vital_signs:
            return None
        return [
            f"혈압은 {vital_signs.get('blood_pressure')}",
            f"맥박은 {vital_signs.get('pulse_rate')}",
            f"호흡수는 {vital_signs.get('respiratory_rate')}",
            f"체온은 {vital_signs.get('temperature')}",
        ]

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


def _python_fallback_format(intent: str, raw_value) -> str:
    """
    Last-resort fallback only.
    Use slight variation so the patient does not sound completely fixed.
    """
    if intent == "greeting":
        return _choose([
            "안녕하세요.",
            "안녕하세요, 선생님.",
        ])

    if intent == "thanks":
        return _choose([
            "네, 감사합니다.",
            "아니에요, 감사합니다.",
        ])

    if intent == "goodbye":
        return _choose([
            "네, 안녕히 계세요.",
            "감사합니다. 안녕히 계세요.",
        ])

    if intent == "small_talk":
        return _choose([
            "네.",
            "네, 괜찮아요.",
            "네, 알겠습니다.",
        ])

    if raw_value is None:
        return _choose([
            "잘 모르겠어요. 다시 말씀해 주시겠어요?",
            "죄송한데 잘 못 알아들었어요. 한 번만 다시 말씀해 주세요.",
        ])

    if intent == "name":
        return _choose([
            f"제 이름은 {raw_value}입니다.",
            f"{raw_value}입니다.",
        ])

    if intent == "age":
        return _choose([
            f"저는 {raw_value}살입니다.",
            f"{raw_value}살이에요.",
        ])

    if intent == "sex":
        return _choose([
            f"{raw_value}입니다.",
            f"{raw_value}예요.",
        ])

    if intent == "chief_complaint":
        return _choose([
            f"{raw_value} 때문에 왔어요.",
            f"{raw_value}가 있어서 왔어요.",
        ])

    if intent == "onset":
        return _choose([
            f"{raw_value}부터 그랬어요.",
            f"{raw_value}쯤부터 아팠어요.",
        ])

    if intent == "location":
        return _choose([
            f"{raw_value}가 아파요.",
            f"{raw_value} 쪽이 아파요.",
        ])

    if intent == "duration":
        return _choose([
            f"{raw_value}.",
            f"{raw_value}됐어요.",
        ])

    if intent == "course":
        return _choose([
            f"{raw_value}.",
            f"시간이 지나면서 {raw_value}.",
        ])

    if intent == "character":
        return _choose([
            f"{raw_value}.",
            f"{raw_value}한 느낌이에요.",
        ])

    if intent == "severity":
        return _choose([
            f"한 {raw_value} 정도예요.",
            f"통증은 {raw_value} 정도 되는 것 같아요.",
        ])

    if intent == "migration":
        return _choose([
            f"{raw_value}.",
            raw_value,
        ])

    if intent == "referred_pain":
        return _choose([
            f"{raw_value}.",
            f"통증이 퍼지는 느낌은 {raw_value}.",
        ])

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
                f"{joined} 같은 증상은 없어요.",
                f"{joined}는 없었어요.",
            ])

        if intent in {
            "past_history",
            "medical_history",
            "social_history",
            "gynecologic_history",
            "vital_signs",
            "aggravating_factors",
            "relieving_factors",
        }:
            return _choose([
                joined,
                f"{joined} 정도예요.",
            ])

        return joined

    return str(raw_value)


def _python_fallback_format_multi(
    primary_intent: str,
    secondary_intent: str | None,
    raw_values: dict,
) -> str:
    if primary_intent == "greeting":
        return _python_fallback_format("greeting", None)

    if primary_intent == "thanks":
        return _python_fallback_format("thanks", None)

    if primary_intent == "goodbye":
        return _python_fallback_format("goodbye", None)

    if primary_intent == "small_talk":
        return _python_fallback_format("small_talk", None)

    if not raw_values:
        return _choose([
            "잘 모르겠어요. 다시 말씀해 주시겠어요?",
            "죄송한데 잘 못 알아들었어요. 다시 말씀해 주세요.",
        ])

    primary_text = _python_fallback_format(primary_intent, raw_values.get(primary_intent))

    if secondary_intent and secondary_intent in raw_values:
        secondary_text = _python_fallback_format(secondary_intent, raw_values.get(secondary_intent))
        if primary_text == secondary_text:
            return primary_text
        return f"{primary_text} {secondary_text}"

    return primary_text


def generate_reply(user_message: str, patient_context: dict, history: list[dict]) -> tuple[str, str]:
    """
    Version 4 architecture:
    1) GPT classifies primary + optional secondary intent
    2) backend deterministically selects the facts
    3) GPT naturalizes the facts into Korean patient speech
    4) Python fallback is used only as a last resort, with mild variation
    """

    intent, secondary_intent = classify_question_intent(user_message, history)
    print(f"[INTENT] {user_message} -> primary={intent}, secondary={secondary_intent}")

    raw_values = _lookup_facts(intent, secondary_intent, user_message, patient_context)
    print(f"[RAW FACTS] {raw_values}")

    if intent == "unknown":
        return _choose([
            "잘 모르겠어요. 다시 말씀해 주시겠어요?",
            "죄송한데 잘 못 알아들었어요. 다시 한번 말씀해 주세요.",
        ]), "unknown"

    try:
        reply = naturalize_patient_answer(
            intent=intent,
            secondary_intent=secondary_intent,
            raw_values=raw_values,
            user_message=user_message,
            patient_context=patient_context,
            history=history,
        )
    except Exception:
        reply = _python_fallback_format_multi(intent, secondary_intent, raw_values)

    if not reply.strip():
        reply = _python_fallback_format_multi(intent, secondary_intent, raw_values)

    combined_intent = f"{intent}+{secondary_intent}" if secondary_intent else intent

    print(f"[FINAL REPLY] {reply}")
    return reply, combined_intent