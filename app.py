from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rvsense_agent import run_rvsense  # must exist

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    session_id: str
    message: str
    is_pro: bool

@app.post("/rvsense")
async def handle_rvsense(payload: ChatInput):
    response = run_rvsense(
        query=payload.message,
        session_id=payload.session_id,
        is_pro=payload.is_pro
    )
    return {"reply": response}
