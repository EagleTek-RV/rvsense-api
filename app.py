from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os

from rvsense_agent import run_rvsense  # Make sure your LangChain logic is in this module

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatInput(BaseModel):
    session_id: str
    message: str
    is_pro: Optional[bool] = False

@app.post("/rvsense")
async def handle_rvsense_chat(payload: ChatInput):
    response = run_rvsense(
        query=payload.message,
        session_id=payload.session_id,
        is_pro=payload.is_pro
    )
    return {"reply": response}
