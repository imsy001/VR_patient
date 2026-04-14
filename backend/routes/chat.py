from fastapi import APIRouter, HTTPException
from backend.models.chat_schema import ChatRequest, ChatResponse
from backend.services.case_loader import load_case
from backend.services.patient_engine import generate_reply

router = APIRouter()


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