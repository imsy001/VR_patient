import json
from pathlib import Path

CASES_DIR = Path(__file__).resolve().parent.parent.parent / "cases"

REQUIRED_FIELDS = [
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
]


def validate_case(case_id: str, case_data: dict) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in case_data]
    if missing:
        raise ValueError(f"Case '{case_id}' missing fields: {', '.join(missing)}")


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