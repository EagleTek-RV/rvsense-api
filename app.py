import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from hashlib import sha256
from flask_cors import CORS

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

app = Flask(__name__)
CORS(app)

session_store = {}
qa_cache = {}

def hash_question(q):
    return sha256(q.lower().strip().encode()).hexdigest()

def get_context(question: str, brand=None, model=None, top_k=5):
    embed = client.embeddings.create(input=question, model="text-embedding-3-small")
    vector = embed.data[0].embedding

    filter_ = {}
    if brand:
        filter_["brand"] = {"$eq": brand}
    if model:
        filter_["model"] = {"$eq": model}

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_ if filter_ else None
    )

    matches = results.matches
    context = "\n\n---\n\n".join([
        f"[{m['metadata'].get('brand', 'Generic')} Manual, pages {m['metadata'].get('page_range', '?')}]: {m['metadata'].get('text', '')}"
        for m in matches if m['score'] > 0.6
    ])
    return context, matches

def ask_gpt(question, context, history=None):
    messages = history if history else []
    if not any(m['role'] == 'system' for m in messages):
        messages = [
            {"role": "system", "content": (
                "You are RVSense, an RV‐repair expert assistant. You have access only to the provided documentation and verified RV manuals (PDFs, Word, Excel, PowerPoint, text) that have been uploaded into your knowledge base. Always quote or reference exactly where your answer comes from (brand, document name, page/section if known). Do not answer from general world knowledge—if the manuals don’t clearly cover the question, reply:\n\nI’m not certain based on the available documentation. Please consult a certified RV technician or the manufacturer’s support line for definitive guidance.\n\nBe concise, exhaustive, and absolutely accurate. Do not hallucinate or guess beyond what’s in the documents."
            )}
        ] + messages

    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages
    )
    return response.choices[0].message.content.strip()

def get_clarifying_question(question):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "The user asked a vague RV tech question. Ask a helpful clarifying follow-up."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "")
    user_id = data.get("user_id", "anon")
    brand = data.get("brand")
    model = data.get("model")
    is_pro = data.get("is_pro", False)

    session = session_store.get(user_id, [])
    qhash = hash_question(question + (brand or '') + (model or ''))

    if qhash in qa_cache:
        return jsonify({"response": qa_cache[qhash]})

    brand_filter = brand if is_pro else "Generic"

    context, matches = get_context(question, brand=brand_filter, model=model)

    if not matches or len(context.strip()) < 50:
        clarifier = get_clarifying_question(question)
        session_store[user_id] = session + [{"role": "assistant", "content": clarifier}]
        return jsonify({"response": clarifier})

    answer = ask_gpt(question, context, history=session)
    session_store[user_id] = session + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    qa_cache[qhash] = answer
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
