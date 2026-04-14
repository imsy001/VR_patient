import json
from pathlib import Path
from typing import Any

CASES_DIR = Path(__file__).resolve().parent.parent.parent / "cases"

# 새 JSON schema 기준 required fields
REQUIRED_TOP_LEVEL_FIELDS = [
    "case_title",
    "patient_info",
    "vital_signs",
    "chief_complaint",
    "history_taking",
    "denied_symptoms",
    "past_history",
]

REQUIRED_PATIENT_INFO_FIELDS = [
    "name",
    "age",
    "sex",
]

REQUIRED_VITAL_SIGNS_FIELDS = [
    "blood_pressure",
    "pulse_rate",
    "respiratory_rate",
    "temperature",
]

REQUIRED_HISTORY_TAKING_FIELDS = [
    "onset",
    "location",
    "duration",
    "course",
    "character",
    "severity",
    "migration",
    "referred_pain",
    "associated_symptoms",
    "aggravating_factors",
    "relieving_factors",
]

REQUIRED_PAST_HISTORY_FIELDS = [
    "trauma",
    "hospitalization",
    "medical_history",
    "medication",
    "social_history",
    "family_history",
    "gynecologic_history",
]

REQUIRED_MEDICAL_HISTORY_FIELDS = [
    "HTN",
    "DM",
    "TB",
    "hepatitis",
]

REQUIRED_SOCIAL_HISTORY_FIELDS = [
    "alcohol",
    "smoking",
    "coffee",
    "occupation",
]

REQUIRED_GYNECOLOGIC_HISTORY_FIELDS = [
    "LMP",
    "menstrual_cycle",
    "pregnancy_possibility",
]


def _require_dict(case_id: str, obj: Any, field_name: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"Case '{case_id}' field '{field_name}' must be a dictionary")
    return obj


def _require_list(case_id: str, obj: Any, field_name: str) -> list:
    if not isinstance(obj, list):
        raise ValueError(f"Case '{case_id}' field '{field_name}' must be a list")
    return obj


def _check_required_fields(case_id: str, data: dict, required_fields: list[str], parent_name: str) -> None:
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(
            f"Case '{case_id}' missing fields in '{parent_name}': {', '.join(missing)}"
        )


def validate_case(case_id: str, case_data: dict) -> None:
    if not isinstance(case_data, dict):
        raise ValueError(f"Case '{case_id}' must be a JSON object")

    # 1. top-level fields
    _check_required_fields(case_id, case_data, REQUIRED_TOP_LEVEL_FIELDS, "root")

    # 2. patient_info
    patient_info = _require_dict(case_id, case_data["patient_info"], "patient_info")
    _check_required_fields(case_id, patient_info, REQUIRED_PATIENT_INFO_FIELDS, "patient_info")

    # 3. vital_signs
    vital_signs = _require_dict(case_id, case_data["vital_signs"], "vital_signs")
    _check_required_fields(case_id, vital_signs, REQUIRED_VITAL_SIGNS_FIELDS, "vital_signs")

    # 4. history_taking
    history_taking = _require_dict(case_id, case_data["history_taking"], "history_taking")
    _check_required_fields(case_id, history_taking, REQUIRED_HISTORY_TAKING_FIELDS, "history_taking")

    _require_list(case_id, history_taking["associated_symptoms"], "history_taking.associated_symptoms")
    _require_list(case_id, history_taking["aggravating_factors"], "history_taking.aggravating_factors")
    _require_list(case_id, history_taking["relieving_factors"], "history_taking.relieving_factors")

    # 5. denied_symptoms
    _require_list(case_id, case_data["denied_symptoms"], "denied_symptoms")

    # 6. past_history
    past_history = _require_dict(case_id, case_data["past_history"], "past_history")
    _check_required_fields(case_id, past_history, REQUIRED_PAST_HISTORY_FIELDS, "past_history")

    medical_history = _require_dict(case_id, past_history["medical_history"], "past_history.medical_history")
    _check_required_fields(
        case_id,
        medical_history,
        REQUIRED_MEDICAL_HISTORY_FIELDS,
        "past_history.medical_history",
    )

    social_history = _require_dict(case_id, past_history["social_history"], "past_history.social_history")
    _check_required_fields(
        case_id,
        social_history,
        REQUIRED_SOCIAL_HISTORY_FIELDS,
        "past_history.social_history",
    )

    gynecologic_history = _require_dict(
        case_id,
        past_history["gynecologic_history"],
        "past_history.gynecologic_history",
    )
    _check_required_fields(
        case_id,
        gynecologic_history,
        REQUIRED_GYNECOLOGIC_HISTORY_FIELDS,
        "past_history.gynecologic_history",
    )


def load_all_cases() -> dict[str, dict]:
    cases: dict[str, dict] = {}

    for path in CASES_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                case_data = json.load(f)

            validate_case(path.stem, case_data)
            cases[path.stem] = case_data

        except Exception as e:
            print(f"[ERROR] Failed to load {path.name}: {e}")

    return cases


def load_case(case_id: str) -> dict:
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Case '{case_id}' not found")

    with open(path, "r", encoding="utf-8") as f:
        case_data = json.load(f)

    validate_case(case_id, case_data)
    return case_data


if __name__ == "__main__":
    loaded = load_all_cases()
    print(f"Loaded {len(loaded)} cases:")
    for case_id, case_data in loaded.items():
        print(f"- {case_id}: {case_data.get('case_title', 'Untitled')}")