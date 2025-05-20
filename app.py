from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from rvsense_agent import run_rvsense

app = FastAPI(
    title="RVSense API",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    session_id: str
    message: str
    is_pro: Optional[bool] = False

@app.get("/")
def health():
    return {"status": "rvsense-api is running"}

@app.post("/rvsense")
async def chat_handler(payload: ChatInput):
    return {
        "reply": run_rvsense(
            query=payload.message,
            session_id=payload.session_id,
            is_pro=payload.is_pro
        )
    }
