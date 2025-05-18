from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load env vars
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pinecone_client.Index(os.getenv("PINECONE_INDEX"))

# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Payload model
class ChatPayload(BaseModel):
    session_id: str
    messages: list
    is_pro: bool = False

def query_pinecone(query_text, is_pro):
    # Embed input
    embedding = openai_client.embeddings.create(
        input=query_text,
        model="text-embedding-3-small"
    ).data[0].embedding

    filter = {} if is_pro else {"access": {"$eq": "free"}}
    results = pinecone_index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True,
        filter=filter
    )
    chunks = [match['metadata']['text'] for match in results.matches]
    return "\n---\n".join(chunks)

@app.post("/chat")
def chat_route(payload: ChatPayload):
    history = payload.messages[-5:]  # optional trimming
    query = history[-1]["content"]

    context = query_pinecone(query, payload.is_pro)

    system_prompt = (
        "You are RVSense, a smart RV tech assistant. Only answer using the provided context. "
        "If you cannot confidently answer, respond: 'This answer may require RVSense Pro.'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context:\n{context}"}
    ] + history

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"OpenAI error: {e}"

    return {"reply": reply}
