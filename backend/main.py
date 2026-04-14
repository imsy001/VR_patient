from fastapi import FastAPI
from backend.routes.chat import router as chat_router

app = FastAPI(title="VR Patient Backend")

app.include_router(chat_router, prefix="/api")


@app.get("/")
def root():
    return {"message": "VR Patient Backend is running"}