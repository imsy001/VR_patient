import random

from backend.services.llm_service import (
    classify_question_intent,
    naturalize_patient_answer,
)


NON_FACT_INTENTS = {"greeting", "thanks", "goodbye", "small_talk", "unknown"}


def _normalize_text_list(values: list[str]) -> set[str]:
    return {str(v).strip().lower() for v in values}


def _match_explicit_denied_symptoms(user_message: str, patient_context: dict) -> list[str]:
    """
    Return a list of explicitly matched denied symptoms from the user's message.
    Do NOT return a fully formed final sentence here.
    """
    lower_msg = user_message.lower()
    denied = _normalize_text_list(patient_context.get("denied_symptoms", []))

    symptom_aliases = {
        "설사": ["설사", "diarrhea"],
        "흉통": ["흉통", "chest pain", "chestpain"],
        "구토": ["구토", "vomiting", "vomit"],
        "발열": ["발열", "fever", "열"],
    }

    matched = []

    for canonical, aliases in symptom_aliases.items():
        if any(alias in lower_msg for alias in aliases):
            if canonical.lower() in denied or any(alias in denied for alias in aliases):
                matched.append(canonical)

    return matched


def _lookup_fact(intent: str, user_message: str, patient_context: dict):
    if intent == "greeting":
        return "__greeting__"

    if intent == "thanks":
        return "__thanks__"

    if intent == "goodbye":
        return "__goodbye__"

    if intent == "small_talk":
        return "__small_talk__"

    if intent == "name":
        return patient_context.get("name")

    if intent == "age":
        return patient_context.get("age")

    if intent == "sex":
        return patient_context.get("sex")

    if intent == "chief_complaint":
        return patient_context.get("chief_complaint")

    if intent == "onset":
        return patient_context.get("onset")

    if intent == "location":
        return patient_context.get("location")

    if intent == "character":
        return patient_context.get("character")

    if intent == "severity":
        return patient_context.get("severity")

    if intent == "associated_symptoms":
        symptoms = patient_context.get("associated_symptoms", [])
        return symptoms if symptoms else None

    if intent == "denied_symptoms":
        explicit_denied = _match_explicit_denied_symptoms(user_message, patient_context)
        if explicit_denied:
            return explicit_denied

        denied = patient_context.get("denied_symptoms", [])
        return denied if denied else None

    if intent == "past_history":
        history_items = patient_context.get("past_history", [])
        return history_items if history_items else None

    return None


def _lookup_facts(
    primary_intent: str,
    secondary_intent: str | None,
    user_message: str,
    patient_context: dict,
) -> dict:
    raw_values = {}

    if primary_intent not in {"unknown"}:
        primary_value = _lookup_fact(primary_intent, user_message, patient_context)
        if primary_value is not None:
            raw_values[primary_intent] = primary_value

    if secondary_intent and secondary_intent not in {"unknown", primary_intent}:
        secondary_value = _lookup_fact(secondary_intent, user_message, patient_context)
        if secondary_value is not None:
            raw_values[secondary_intent] = secondary_value

    return raw_values


def _choose(options: list[str]) -> str:
    return random.choice(options)


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

    if intent == "character":
        return _choose([
            f"{raw_value}하게 아파요.",
            f"{raw_value}한 느낌으로 아파요.",
        ])

    if intent == "severity":
        return _choose([
            f"한 {raw_value} 정도예요.",
            f"통증은 {raw_value} 정도 되는 것 같아요.",
        ])

    if isinstance(raw_value, str):
        return raw_value

    if isinstance(raw_value, list):
        if intent == "associated_symptoms":
            joined = ", ".join(raw_value)
            return _choose([
                f"{joined}도 있어요.",
                f"{joined} 같은 증상도 있었어요.",
            ])

        if intent == "denied_symptoms":
            joined = ", ".join(raw_value)
            return _choose([
                f"{joined} 같은 증상은 없어요.",
                f"{joined}는 없었어요.",
            ])

        if intent == "past_history":
            joined = ", ".join(raw_value)
            return _choose([
                joined,
                f"{joined} 정도 있었어요.",
            ])

        return ", ".join(raw_value)

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