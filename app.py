from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from rvsense_agent import run_rvsense  # This must match your LangChain module

app = FastAPI()

# CORS setup — allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your actual domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class ChatInput(BaseModel):
    session_id: str
    message: str
    is_pro: Optional[bool] = False  # must be included

@app.get("/")
def health_check():
    return {"status": "rvsense-api running"}

@app.post("/rvsense")
async def handle_chat(payload: ChatInput):
    reply = run_rvsense(
        query=payload.message,
        session_id=payload.session_id,
        is_pro=payload.is_pro
    )
    return {"reply": reply}
