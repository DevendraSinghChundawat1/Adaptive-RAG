from vectore_store import index_documents
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from retrieval_grader import grade_retrieved_document
from generate import generate_llm_response
from hallucination_grader import grade_hallucination
from answer_grader import grade_answer
from  router import route_question
from langgraph.graph import END, StateGraph, START
from define_langgraph import *
from langchain_community.tools.tavily_search import TavilySearchResults

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["USER_AGENT"] = "DevendraSingh"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

web_search_tool = TavilySearchResults(k=3)

local_llm = "llama3.1:8b"


# Example usage
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

retriever = index_documents(urls, vector_db_dir="chroma_vector_db")


llm = ChatOllama(model=local_llm, temperature=0)
retrieval_grader = grade_retrieved_document(llm)
rag_chain = generate_llm_response(llm)
hallucination_grader = grade_hallucination(llm)
answer_grader = grade_answer(llm)
question_router = route_question(llm)


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae



# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

# Compile
app = workflow.compile()

# Test

inputs = {"question": "What are the types of agent memory?"}
for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["generation"])