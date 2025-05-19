from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import logging
import json

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pinecone_client.Index(os.getenv("PINECONE_INDEX"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

class Message(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    session_id: str
    messages: list[Message]
    is_pro: bool = False
    brand: str | None = None
    model: str | None = None
    year: int | None = None

def dedupe_messages(messages):
    seen = set()
    deduped = []
    for msg in messages:
        sig = (msg.role, msg.content.strip())
        if sig not in seen:
            seen.add(sig)
            deduped.append(msg)
    return deduped[-10:]

def query_pinecone(query_text, is_pro, brand=None, model=None, year=None):
    embedding = openai_client.embeddings.create(
        input=query_text,
        model="text-embedding-3-small"
    ).data[0].embedding

    filter = {} if is_pro else {"access": {"$eq": "free"}}
    if brand:
        filter["brand"] = {"$eq": brand}
    if model:
        filter["model"] = {"$eq": model}
    if year:
        filter["year"] = {"$eq": year}

    results = pinecone_index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True,
        filter=filter
    )
    chunks = [
        f"{m['metadata'].get('text', '')}\n\n(Source: {m['metadata'].get('citation', 'Unknown')}, Page: {m['metadata'].get('page', '?')})"
        for m in results.matches
    ]
    return "\n---\n".join(chunks)

@app.post("/chat")
def chat_route(payload: ChatPayload):
    history = dedupe_messages(payload.messages[-10:])
    query = history[-1].content if history else ""

    context = query_pinecone(query, payload.is_pro, payload.brand, payload.model, payload.year)

    system_prompt = (
        "You are RVSense Pro, a smart assistant for diagnosing and explaining RV systems.\n\n"
        "Your knowledge comes entirely from technical manuals, user guides, and installation instructions. Use the metadata from the matched document chunks to guide your answers.\n\n"
        "Rules:\n"
        "- Only answer based on provided chunks.\n"
        "- Prioritize chunks with a high confidence_score.\n"
        "- Use make, model, and year to filter relevant chunks. Do not guess across models or brands.\n"
        "- Refer to the section_title and manual_section to understand context.\n"
        "- If images are referenced in the metadata, mention their filenames in your answer.\n"
        "- Cite the source of the information using citation and source_url.\n"
        "- If you cannot find the answer in the data provided, say: \"This information isn’t in the manual data I have access to.\"\n\n"
        "Example citation format:\n"
        "\"According to the 2020 Forest River Georgetown GT5 Owner's Manual, page 23...\"\n\n"
        "If the user provides partial info (e.g., just a model), ask clarifying questions to narrow the result."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Relevant manual excerpts:\n{context}"}
    ] + [msg.dict() for msg in history]

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

    logging.info(json.dumps({
        "session_id": payload.session_id,
        "query": query,
        "context_used": context,
        "response": reply
    }))

    return {"reply": reply}
