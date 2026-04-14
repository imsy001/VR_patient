from backend.services.case_loader import load_all_cases

if __name__ == "__main__":
    cases = load_all_cases()
    print(f"Loaded {len(cases)} cases:")
    for case_id, case_data in cases.items():
        print(f"- {case_id}: {case_data.get('case_title', 'Untitled')}")