from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
import os
from dotenv import load_dotenv
from typing import Dict

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
embeddings = OpenAIEmbeddings()

# Global memory store by session
memory_store: Dict[str, ConversationBufferMemory] = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    return memory_store[session_id]

def get_retriever(is_pro: bool):
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embeddings
    )
    filters = {} if is_pro else {"access": {"$eq": "free"}}
    return vectorstore.as_retriever(search_kwargs={"k": 5, "filter": filters})

# Example tools

def lookup_part_number(query: str) -> str:
    return f"Part number for '{query}' is SUB-4567"

def schedule_tech_visit(details: str) -> str:
    return f"Tech scheduled for: {details}"

tools = [
    Tool(
        name="PartsLookup",
        func=lookup_part_number,
        description="Lookup part numbers for RV components"
    ),
    Tool(
        name="ServiceScheduler",
        func=schedule_tech_visit,
        description="Schedule a service appointment for the RV"
    )
]

# Core router

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["schedule", "appointment"]):
        return "schedule"
    if any(k in q for k in ["part number", "replace", "anode", "thermostat"]):
        return "parts"
    return "qa"

def run_rvsense(query: str, session_id: str, is_pro: bool) -> str:
    memory = get_memory(session_id)
    intent = detect_intent(query)

    if intent in ["schedule", "parts"]:
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            memory=memory
        )
        return agent.run(query)
    else:
        retriever = get_retriever(is_pro)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )
        result = qa_chain({"question": query, "chat_history": []})
        sources = [doc.metadata.get("source", "") for doc in result["source_documents"]]
        return f"{result['answer']}\n\nSources:\n" + "\n".join(sources)

# Example test loop (remove in production)
if __name__ == "__main__":
    session = "test-session"
    while True:
        user_input = input("You: ")
        response = run_rvsense(user_input, session_id=session, is_pro=False)
        print(f"RVSense: {response}")
