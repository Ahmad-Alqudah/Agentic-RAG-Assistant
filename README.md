# Agentic RAG Assistant

An end-to-end, locally hosted Agentic Retrieval-Augmented Generation
system built using LangGraph, hybrid retrieval, Ollama, FastAPI,
Streamlit, and Docker.

## Overview

This project demonstrates an AI assistant that retrieves information
from company documents, evaluates available evidence, rewrites queries
when retrieval is insufficient, and selects an appropriate workflow.

The system supports three outcomes:

- Answered: generate a grounded response.
- Abstained: refuse to invent missing information.
- Human Review: flag sensitive requests for manual review.

## Technology Stack

- Python 3.11
- LangGraph
- LangChain Tools
- Sentence Transformers
- FAISS
- BM25
- Reciprocal Rank Fusion (RRF)
- Ollama
- Qwen2.5 0.5B
- FastAPI
- Streamlit
- Pytest
- Docker Compose

## Architecture

User Question
    |
    v
Hybrid Retrieval (FAISS + BM25)
    |
    v
Evidence Evaluation
    |
    +--> Insufficient Evidence
    |         |
    |         v
    |    Query Rewriting
    |         |
    |         v
    |    Retrieval Retry (maximum 1)
    |
    +--> Sensitive Request
    |         |
    |         v
    |    Human Review Flag
    |
    +--> Relevant Evidence
              |
              v
         Ollama / Qwen
              |
              v
         Answer Validation
              |
              +--> Answered
              |
              +--> Abstained

## Features

### Hybrid Retrieval

Uses FAISS semantic search and BM25 keyword search.
Results are combined using Reciprocal Rank Fusion.

### Agentic Workflow

LangGraph controls retrieval, evidence evaluation,
query rewriting, retries, generation, and routing.

### Query Rewriting

Rule-based query rewriting is used when the initial
retrieval is insufficient. The retry limit is one.

### Local LLM

Qwen2.5 0.5B runs locally through Ollama.
No paid LLM API is required.

### Answer Validation

A lightweight lexical-overlap check filters some
unsupported generated responses.

This check is not semantic fact verification.

### Human Review

Sensitive requests are routed to a human-review status.

The current implementation flags requests for review;
it does not implement an interactive approval workflow.

## Project Structure

```text
03_Agentic_RAG_Assistant/
├── app/
│   ├── agents/
│   │   └── agent_graph.py
│   ├── api/
│   │   └── main.py
│   └── tools/
│       └── document_search.py
├── data/
│   └── documents/
├── evaluation/
│   ├── final_evaluation.py
│   └── results.json
├── frontend/
│   └── streamlit_app.py
├── tests/
│   ├── conftest.py
│   └── test_agent_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and run Ollama with:

```bash
ollama pull qwen2.5:0.5b
```

Start FastAPI:

```bash
python -m uvicorn app.api.main:app --port 8003
```

Start Streamlit:

```bash
python -m streamlit run frontend/streamlit_app.py --server.port 8504
```

## Running with Docker

Start Ollama on the host machine.

Then run:

```bash
docker compose up --build -d
```

Access:

- FastAPI: http://localhost:8004/docs
- Streamlit: http://localhost:8505
- Health: http://localhost:8004/health

Docker connects to host Ollama using
`host.docker.internal:11434`.

## Automated Tests

```bash
python -m pytest tests/test_agent_api.py -v
```

Observed result:

**10/10 API tests passed (mocked agent responses).**

## Final Evaluation

Run the real agent evaluation:

```bash
python -m evaluation.final_evaluation
```

The evaluation contains nine synthetic test cases:

- Three answerable questions
- Three unanswerable questions
- Three human-review questions

Results are saved to `evaluation/results.json`.

Record the measured results after running the evaluation.
Do not treat API mock tests as LLM accuracy measurements.

## Known Limitations

- Small synthetic document collection.
- Rule-based query rewriting.
- Lexical evidence evaluation.
- Lexical answer validation.
- Human review is a flag, not an approval system.
- Qwen2.5 0.5B has limited reasoning capacity.
- Retrieved sources are not independently verified citations.
- No authentication, authorization, or persistent review queue.
- No production-grade observability or scaling.

## Future Improvements

- Semantic evidence grading.
- Claim-level citation verification.
- LangGraph interrupts and persistent checkpoints.
- Human approval and rejection workflow.
- Improved query rewriting using a stronger LLM.
- Authentication, monitoring, and production deployment.

## Portfolio Summary

Built a locally hosted Agentic RAG assistant using LangGraph,
hybrid FAISS/BM25 retrieval, query rewriting, bounded retries,
Qwen2.5 via Ollama, FastAPI, Streamlit, and Docker Compose.

Implemented evidence-based routing, abstention,
lightweight answer validation, and human-review flagging.