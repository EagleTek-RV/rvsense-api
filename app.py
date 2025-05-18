from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import pinecone
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")

# Init Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(pinecone_index_name)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class ChatPayload(BaseModel):
    session_id: str
    messages: list
    is_pro: bool = False

# Helper: Query Pinecone
def query_pinecone(query_text, is_pro):
    vector = openai.Embedding.create(
        input=query_text,
        model="text-embedding-3-small"
    )["data"][0]["embedding"]

    filters = {"access": {"$eq": "free"}} if not is_pro else {}

    results = index.query(
        vector=vector,
        top_k=5,
        include_metadata=True,
        filter=filters
    )

    contexts = [match["metadata"]["text"] for match in results["matches"]]
    return "\n---\n".join(contexts)

# Route: Chat endpoint
@app.post("/chat")
async def chat_handler(payload: ChatPayload):
    user_messages = payload.messages
    session_id = payload.session_id
    is_pro = payload.is_pro

    latest_question = user_messages[-1]["content"]

    # Get context from Pinecone
    context = query_pinecone(latest_question, is_pro)

    system_prompt = (
        "You are RVSense, a world-class RV assistant. "
        "Answer ONLY based on the provided documentation. "
        "If you don't have enough information, say 'This answer may require RVSense Pro.'"
    )

    gpt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{context}"}
    ] + user_messages[-5:]  # include last 5 messages for continuity

    # Query OpenAI
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=gpt_messages,
            temperature=0.2,
            max_tokens=800
        )
        assistant_reply = completion.choices[0].message["content"]
    except Exception as e:
        return {"reply": f"OpenAI error: {str(e)}"}

    return {"reply": assistant_reply}
