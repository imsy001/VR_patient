from fastapi import APIRouter, HTTPException
from backend.models.chat_schema import ChatRequest, ChatResponse
from backend.services.case_loader import load_case, load_all_cases
from backend.services.patient_engine import generate_reply

router = APIRouter()


@router.get("/cases")
def get_cases():
    cases = load_all_cases()

    result = []
    for case_id, case_data in cases.items():
        patient_info = case_data.get("patient_info", {})

        result.append({
            "case_id": case_id,
            "case_title": case_data.get("case_title", "Untitled"),
            "name": patient_info.get("name", ""),
            "age": patient_info.get("age", ""),
            "sex": patient_info.get("sex", ""),
            "chief_complaint": case_data.get("chief_complaint", ""),
        })

    return result


@router.get("/case/{case_id}")
def get_case(case_id: str):
    try:
        return load_case(case_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Case not found")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        patient_context = load_case(request.case_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Case not found")

    reply, intent = generate_reply(
        user_message=request.message,
        patient_context=patient_context,
        history=[item.model_dump() for item in request.history],
    )

    return ChatResponse(reply=reply, intent=intent)