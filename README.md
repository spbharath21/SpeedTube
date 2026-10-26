# ⚡ SpeedTube

[![CI](https://github.com/divijaiwanth/SpeedTube/actions/workflows/ci.yml/badge.svg)](https://github.com/divijaiwanth/SpeedTube/actions/workflows/ci.yml)

> Get up to speed on any YouTube video — instantly.

SpeedTube is a production-grade RAG (Retrieval-Augmented Generation) system that lets you ask questions, get summaries, and extract insights from any YouTube video using its transcript. Built with advanced retrieval techniques, a FastAPI backend, a React frontend, and production guardrails.

---

## What it does

Paste any YouTube URL. SpeedTube fetches the transcript, builds a semantic search index, and lets you have a conversation with the video — asking questions, getting summaries, or pulling key quotes — all without watching a single second of it.

---

## Architecture

```
YouTube URL
    ↓
Transcript fetch (youtube-transcript-api)
    ↓
Parent-child chunking (LangChain)
    ↓
┌─────────────────────────┐
│   Dual Index            │
│   FAISS (semantic)      │
│   BM25  (keyword)       │
└─────────────────────────┘
    ↓
Query comes in
    ↓
Input Guardrails (validate question)
    ↓
Circuit Breaker (check Groq health)
    ↓
HyDE — generate hypothetical answer, embed it
    ↓
Hybrid retrieval — RRF fusion of FAISS + BM25
    ↓
Cross-encoder reranking (BAAI/bge-reranker-base)
    ↓
Parent chunk lookup — fetch full context
    ↓
LLM answer generation (Groq / llama-3.1-8b)
    ↓
FastAPI response → React UI
```

---

## Tech Stack

| Layer | Tech |
|---|---|
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Reranker | BAAI/bge-reranker-base (cross-encoder) |
| Vector store | FAISS |
| Keyword search | BM25 (rank-bm25) |
| RAG framework | LangChain |
| Backend | FastAPI + Uvicorn |
| Frontend | React |
| Cache | Redis |
| Rate limiting | slowapi |
| Logging | loguru |
| Evaluation | RAGAS + MLflow |

---

## Project Structure

```
SpeedTube/
├── .github/workflows/ci.yml  # Lint (ruff) + frontend build on every push/PR
├── backend/
│   ├── main.py               # FastAPI app — all API endpoints + guardrails
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.js            # React app — UI components
│   │   └── App.css           # Styles
│   └── Dockerfile
├── docker-compose.yml         # backend + frontend + redis, one command
├── langchain_helper.py       # Core RAG pipeline (Phase 1)
├── bedrock_helper.py         # AWS Bedrock embeddings + LLM (Phase 5)
├── evaluate.py                # RAGAS evaluation pipeline (Phase 2)
├── cookies.txt                # Optional — YouTube session cookies (gitignored)
├── logs/                      # Runtime logs (gitignored)
├── pyproject.toml / uv.lock
└── .env                       # API keys (never committed)
```

---

## API Endpoints

Full interactive docs at `http://localhost:8000/docs`.

### `POST /ingest`
Load a YouTube video and build the RAG index.
Rate limited to **5 requests/minute** per IP. Idempotent.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=zjkBMFhNj_g"}'
```

```json
{
  "session_id": "92de3076a6db",
  "video_id": "zjkBMFhNj_g",
  "chunk_count": 320,
  "message": "Video loaded successfully",
  "cached": false
}
```

### `POST /query`
Ask a question about a loaded video.
Rate limited to **10 requests/minute** per IP.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "92de3076a6db", "question": "What is a large language model?"}'
```

```json
{
  "answer": "A large language model is...",
  "sources": ["...chunk 1...", "...chunk 2...", "...chunk 3..."],
  "num_sources": 3
}
```

### `GET /summary`
Get a summary of a loaded video.
Rate limited to **5 requests/minute** per IP.

```bash
curl "http://localhost:8000/summary?session_id=92de3076a6db&summary_type=bullets"
```

`summary_type` options: `concise`, `detailed`, `bullets`

### `GET /health`
Returns Redis status, circuit breaker state, and uptime.

```json
{
  "status": "ok",
  "redis_connected": true,
  "circuit_breaker_state": "CLOSED",
  "uptime_seconds": 142.3
}
```

### `DELETE /session/{session_id}`
Clear a session from cache to free memory.

---

## Phase 1 — Advanced RAG

### Parent-Child Chunking
Small chunks (300 chars) are used for retrieval — precise matching. When a child chunk is retrieved, its parent (1200 chars) is sent to the LLM — full context. Prevents relevant context from being split across chunk boundaries.

### Hybrid Search + RRF
BM25 handles exact keyword matches (names, technical terms). FAISS handles semantic similarity. Both run in parallel and are fused using Reciprocal Rank Fusion:

```
score = Σ 1 / (rank + 60)
```

### HyDE (Hypothetical Document Embeddings)
Instead of embedding the raw question, the LLM generates a hypothetical answer first, then that answer is embedded. A hypothetical answer is lexically closer to the actual transcript than a short question, improving cosine similarity matching significantly.

Paper: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)

### Cross-Encoder Reranking
10 candidate chunks are retrieved via hybrid search. All 10 are passed through `BAAI/bge-reranker-base` — a cross-encoder that scores (query, document) pairs jointly. Top 3 are kept. Two-stage approach: FAISS for speed, cross-encoder for accuracy.

---

## Phase 2 — Evaluation

### Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Are all claims in the answer supported by the retrieved context? |
| Answer Relevancy | Is the answer actually relevant to the question asked? |
| Context Precision | Are relevant chunks ranked before irrelevant ones? |
| Context Recall | Did retrieval find all the chunks needed to answer? |

### Running evaluation

```bash
python evaluate.py <youtube_video_id> <n_questions>
python evaluate.py zjkBMFhNj_g 10
```

### Viewing results in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

### Sample results

| Video type | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Conversational | 0.25 | 0.82 | 0.00 | 0.00 |
| Technical tutorial | 0.33 | 0.56 | 0.40 | 0.50 |

---

## Phase 4 — Production Guardrails

### Rate Limiting
Built with `slowapi` — a FastAPI-compatible rate limiting library using the token bucket algorithm.

| Endpoint | Limit |
|---|---|
| `POST /ingest` | 5 requests/minute |
| `POST /query` | 10 requests/minute |
| `GET /summary` | 5 requests/minute |

Clients that exceed the limit receive a `429 Too Many Requests` response automatically.

### Redis Caching
Sessions are cached in Redis with a 3-hour TTL (sliding expiry — resets on each access).

```
First call  → build FAISS index (10-30 seconds)
Second call → Redis cache hit (< 100ms)
```

The cache is persistent — sessions survive server restarts. Falls back to in-memory if Redis is unavailable.

### Input Guardrails
Every question is validated before hitting the LLM:

- Empty questions rejected
- Questions under 5 characters rejected
- Questions over 500 characters rejected
- Prompt injection patterns blocked (`"ignore previous instructions"`, `"jailbreak"` etc.)
- Non-YouTube URLs rejected at `/ingest`

### Circuit Breaker
Protects against cascading failures when Groq API is down.

```
State: CLOSED → normal operation
         ↓ (3 consecutive failures)
State: OPEN → all requests blocked, returns 503 immediately
         ↓ (60 second cooldown)
State: HALF-OPEN → one test request allowed
         ↓ (success)
State: CLOSED → normal operation resumes
```

The circuit breaker state is visible at `GET /health`.

### Structured Logging
All logs written via `loguru` to both terminal and `logs/speedtube.log`.

```
2026-06-13 11:04:26 | INFO | api:89 | Cache HIT for video zjkBMFhNj_g
2026-06-13 11:04:31 | WARNING | api:134 | Invalid question rejected: too short
2026-06-13 11:04:45 | WARNING | api:112 | Circuit breaker OPENED after 3 failures
```

Log files rotate at 10MB and are retained for 7 days.

---

## Setup

### Option A — Docker Compose (recommended)

Spins up the backend, frontend, and Redis together with one command.

```bash
git clone https://github.com/divijaiwanth/SpeedTube.git
cd SpeedTube

# Add your API keys
cp .env_template .env   # then fill in GROQ_API_KEY

docker compose up --build
```

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000` (docs at `/docs`)

First run downloads the embedding + reranker models from HuggingFace (~1-3 minutes); they're cached in a named Docker volume, so every run after that starts instantly.

### Option B — Local (no Docker)

#### Prerequisites
- Python 3.11+
- Node.js 18+
- Redis is optional; the API falls back to an in-memory session cache
- A [Groq API key](https://console.groq.com) (free)

#### Backend

```bash
git clone https://github.com/spbharath21/SpeedTube.git
cd SpeedTube

uv sync
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Set environment variables
$env:GROQ_API_KEY = "gsk_your_key_here"   # Windows
export GROQ_API_KEY="gsk_your_key_here"   # Mac/Linux

# Run backend from the repository root
uv run uvicorn backend.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm start
```

---

## Roadmap

- [x] Phase 1 — Advanced RAG pipeline (hybrid search, reranking, HyDE, parent-child chunking)
- [x] Phase 2 — RAGAS evaluation + MLflow experiment tracking
- [x] Phase 3 — FastAPI backend + React frontend
- [x] Phase 4 — Rate limiting, Redis caching, guardrails, circuit breaker, structured logging
- [x] Phase 5 — AWS Bedrock migration (Titan + Nova Micro)
- [x] Containerization — Dockerfiles for backend/frontend + docker-compose
- [x] CI — ruff lint/format + frontend build on every push/PR ([workflow](.github/workflows/ci.yml))

Optional future work: deploy the existing Bedrock backend through AWS-managed infrastructure.

---

## Known Limitations

- Context precision and recall metrics score 0 on conversational videos with llama-3.1-8b as judge — will be resolved in Phase 5 with a stronger Bedrock model.
- YouTube occasionally rate-limits transcript requests from unauthenticated/cloud IPs (`429`). Drop a `cookies.txt` (Netscape format, exported from a logged-in browser session) in the repo root — `langchain_helper.py` picks it up automatically if present.
- Circuit breaker state resets on server restart — persistent state coming with Redis in Phase 5.
