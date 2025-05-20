from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Initialize Pinecone Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX,
    embedding=embeddings
)

# Setup memory for session persistence (should be session-specific in production)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define retriever chain with optional metadata filter for access control
retriever = vectorstore.as_retriever(search_kwargs={
    "k": 5,
    "filter": {"access": {"$eq": "free"}}  # Replace or expand based on user tier (e.g., free vs pro)
})
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# Example external tools (replace with real implementations)
def lookup_part_number(query: str) -> str:
    return f"Part number for '{query}' is SUB-4567"

def schedule_tech_visit(details: str) -> str:
    return f"Tech scheduled for: {details}"

# Register tools for the agent
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

# Initialize agent with tools and chat LLM
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Main RVSense interface

def run_rvsense(query: str):
    normalized_query = query.lower()
    if any(keyword in normalized_query for keyword in ["schedule", "appointment"]):
        return agent.run(query)
    elif any(keyword in normalized_query for keyword in ["part number", "replace", "anode", "thermostat"]):
        return agent.run(query)
    else:
        result = retrieval_chain.run(query)
        # Future: add citation or confidence post-processing here
        return result

# Example test (remove or refactor for production API)
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        response = run_rvsense(user_input)
        print(f"RVSense: {response}")
