import os
from typing import TypedDict
import re
import requests

from langgraph.graph import StateGraph, START, END

from app.tools.document_search import get_engine


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = "qwen2.5:0.5b"

ABSTENTION = (
    "I cannot find the answer in the provided documents."
)


class AgentState(TypedDict):
    question: str
    search_query: str
    retrieved_context: str
    retrieved_sources: list[str]
    relevant: bool
    answer: str
    route: str
    retry_count: int
    status: str
    validation_passed: bool


REWRITE_RULES = {
    "work from home": "remote work",
    "working from home": "remote work",
    "login problem": "account access issue",
    "sign in problem": "account access issue",
    "critical ticket": "P1 production outage",
    "deploy application": "production deployment",
}


def rewrite_query(question: str) -> str:
    rewritten = question.lower().strip()
    for old, new in REWRITE_RULES.items():
        rewritten = rewritten.replace(old, new)
    return rewritten


def retrieve_node(state: AgentState):
    results = get_engine().search(
        state["search_query"],
        top_k=3,
    )

    context = "\n\n".join(
        f"Source: {item['source']}\nContent: {item['text']}"
        for item in results
    )

    return {
        "retrieved_context": context,
        "retrieved_sources": [
            item["source"] for item in results
        ],
    }


def normalize_token(token: str) -> str:
    token = token.lower()

    # Simple normalization for common verb endings.
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    if token.endswith("es") and len(token) > 4:
        return token[:-1]

    if token.endswith("s") and len(token) > 4:
        return token[:-1]

    return token


def evaluate_node(state: AgentState):
    engine = get_engine()

    stopwords = {
        "what", "does", "mean", "the", "is",
        "a", "an", "of", "in", "who", "are",
        "how", "can", "i", "to", "for",
        "about", "tell", "me", "please",
    }

    query_tokens = {
        normalize_token(token)
        for token in engine.tokenize(state["search_query"])
        if token not in stopwords
    }

    context_tokens = {
        normalize_token(token)
        for token in engine.tokenize(
            state["retrieved_context"]
        )
    }

    relevant = bool(query_tokens) and (
        query_tokens.issubset(context_tokens)
    )

    if needs_human_review(state["question"]):
        route = "human_review"
    elif relevant:
        route = "generate"
    elif state["retry_count"] < 1:
        route = "rewrite"
    else:
        route = "abstain"

    return {
        "relevant": relevant,
        "route": route,
    }

def needs_human_review(question: str) -> bool:
    sensitive_terms = [
        "unauthorized access",
        "security incident",
        "password reset",
        "disable mfa",
        "phishing",
    ]

    text = question.lower()

    return any(
        term in text
        for term in sensitive_terms
    )


def rewrite_node(state: AgentState):
    return {
        "search_query": rewrite_query(state["question"]),
        "retry_count": state["retry_count"] + 1,
    }


def generate_node(state: AgentState):
    prompt = f"""You are a company document assistant.

Answer the question using ONLY the context below.
Do not invent information.
If the answer is missing, respond exactly:
{ABSTENTION}

Context:
{state['retrieved_context']}

Question:
{state['question']}

Answer:"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            },
        },
        timeout=180,
    )

    response.raise_for_status()

    answer = response.json()["response"].strip()

    if answer.startswith(ABSTENTION):
        answer = ABSTENTION

    return {"answer": answer}


def validate_node(state: AgentState):
    answer = state["answer"]

    if not answer or answer == ABSTENTION:
        return {
            "validation_passed": False,
            "status": "abstained",
            "answer": ABSTENTION,
        }

    context_tokens = set(
        re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)*",
            state["retrieved_context"].lower(),
        )
    )

    answer_tokens = set(
        re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)*",
            answer.lower(),
        )
    )

    stopwords = {
        "the", "a", "an", "is", "are",
        "to", "of", "and", "in", "for",
        "it", "that", "this", "with",
    }

    meaningful_tokens = answer_tokens - stopwords

    overlap = (
        len(meaningful_tokens & context_tokens)
        / max(len(meaningful_tokens), 1)
    )

    passed = overlap >= 0.5

    return {
        "validation_passed": passed,
        "status": "answered" if passed else "abstained",
        "answer": answer if passed else ABSTENTION,
    }


def human_review_node(state: AgentState):
    return {
        "status": "human_review",
        "answer": (
            "This request requires human review "
            "before an answer can be provided."
        ),
        "validation_passed": False,
    }


def abstain_node(state: AgentState):
    return {
        "status": "abstained",
        "answer": ABSTENTION,
        "validation_passed": False,
    }


def route_after_evaluation(state: AgentState):
    return state["route"]


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("abstain", abstain_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "human_review": "human_review",
            "abstain": "abstain",
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", END)
    workflow.add_edge("human_review", END)
    workflow.add_edge("abstain", END)

    return workflow.compile()


def run_agent(question: str):
    graph = build_graph()

    return graph.invoke({
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
    })


if __name__ == "__main__":
    questions = [
        "What is the work from home policy?",
        "Who is the CEO of NovaTech?",
        "What should I do about a security incident?",
    ]

    for question in questions:
        print("\n" + "=" * 55)
        print("QUESTION:", question)

        try:
            result = run_agent(question)

            print("STATUS:", result["status"])
            print("RETRIES:", result["retry_count"])
            print("VALIDATED:", result["validation_passed"])
            print("ANSWER:", result["answer"])
            print("SOURCES:", result["retrieved_sources"])

        except Exception as error:
            print("ERROR:", repr(error))