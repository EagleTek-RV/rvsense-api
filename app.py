from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from rvsense_agent import run_rvsense  # Your LangChain logic here

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
    is_pro: Optional[bool] = False

@app.post("/debug")
async def debug_echo(request: Request):
    return await request.json()

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/rvsense")
async def chat_handler(payload: ChatInput):
    return {"reply": run_rvsense(payload.message, payload.session_id, payload.is_pro)}
