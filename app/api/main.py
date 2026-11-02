from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agents.agent_graph import build_graph
from app.tools.document_search import get_engine


graph = None
graph_lock = Lock()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AgentResponse(BaseModel):
    question: str
    answer: str
    status: str
    route: str
    retry_count: int
    validation_passed: bool
    sources: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph

    # Load embeddings and retrieval index once.
    get_engine()

    # Compile LangGraph once.
    graph = build_graph()

    yield

    graph = None


app = FastAPI(
    title="Agentic RAG Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "Agentic RAG Assistant",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if graph is not None else "starting",
        "agent_loaded": graph is not None,
    }


@app.post("/ask", response_model=AgentResponse)
def ask(request: QuestionRequest):

    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Agent is not ready.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    initial_state = {
        "question": question,
        "search_query": question,
        "retrieved_context": "",
        "retrieved_sources": [],
        "relevant": False,
        "answer": "",
        "route": "",
        "retry_count": 0,
        "status": "",
        "validation_passed": False,
    }

    try:
        # Serialize requests around the shared graph.
        with graph_lock:
            result = graph.invoke(initial_state)

    except Exception as error:
        print(f"Agent execution failed: {error!r}")
        raise HTTPException(
            status_code=503,
            detail="Agent execution failed. Check server logs.",
        ) from error

    # Do not expose retrieved documents as supporting
    # citations when the answer was not validated.
    sources = (
        result["retrieved_sources"]
        if result["status"] == "answered"
        else []
    )

    return AgentResponse(
        question=question,
        answer=result["answer"],
        status=result["status"],
        route=result["route"],
        retry_count=result["retry_count"],
        validation_passed=result["validation_passed"],
        sources=sources,
    )