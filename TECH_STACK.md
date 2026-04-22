# FORGE Tech Stack

A complete overview of the technologies, frameworks, and tools used in the FORGE (Forensic RL Graph Environment) project.

---

## Language

| Technology | Version | Purpose |
|---|---|---|
| **Python** | ≥ 3.11 | Primary programming language |

---

## Core ML / Reinforcement Learning

| Technology | Version | Purpose |
|---|---|---|
| **PyTorch** | ≥ 2.1.0 | Deep learning framework for neural network policy (PPO, GNN) |
| **Gymnasium** | ≥ 0.29.1 | Standard RL environment interface (OpenAI Gym-compatible) |
| **NumPy** | ≥ 1.24.0 | Numerical computing and array operations |
| **sentence-transformers** | ≥ 2.6.0 | Local text embeddings — `all-MiniLM-L6-v2` (384-dim, runs fully offline) |
| **PyTorch Geometric** *(optional)* | ≥ 2.5.0 | Graph Neural Network (GNN) policy; requires GPU |
| **Ray RLlib** *(optional)* | ≥ 2.9.0 | Distributed PPO training across multiple workers |

---

## Web Framework / REST API

| Technology | Version | Purpose |
|---|---|---|
| **FastAPI** | ≥ 0.111.0 | Async REST API server exposing the OpenEnv-compatible `/reset`, `/step`, `/state` endpoints |
| **Uvicorn** | ≥ 0.29.0 | ASGI server that runs the FastAPI application |
| **Pydantic** | ≥ 2.0.0 | Request/response data validation and serialisation |

---

## Frontend / UI

| Technology | Version | Purpose |
|---|---|---|
| **Gradio** | ≥ 4.0.0 | Web-based interactive UI (`app.py`) for demonstrating investigations in the browser |

---

## LLM Integration

| Technology | Details | Purpose |
|---|---|---|
| **OpenAI Python client** | ≥ 1.14.0 | OpenAI-compatible API client (pointed at Groq for free-tier LLM access) |
| **Groq API** | `llama3-8b-8192` (default) | Free-tier LLM inference endpoint — drop-in replacement for OpenAI API |
| **Tenacity** | ≥ 8.2.0 | Exponential-backoff retry logic for Groq 429 rate-limit errors |

---

## External / Free APIs

| API | Purpose |
|---|---|
| **Wikipedia REST API** | Primary source verification and article summaries |
| **DuckDuckGo Search** (`duckduckgo-search`) | Keyless, completely free fact-check web search |
| **Wayback Machine (Internet Archive)** | Timestamp verification and backdated-article detection |
| **Wikidata SPARQL** | Entity disambiguation and knowledge-graph lookups |
| **MediaStack API** *(optional, 500 req/month free)* | Real-world news article retrieval |
| **GNews API** *(optional, 100 req/day free)* | Additional news source cross-referencing |

---

## HTTP Client

| Technology | Version | Purpose |
|---|---|---|
| **httpx** | ≥ 0.27.0 | Async HTTP client for all external API calls inside forensic tools |

---

## Configuration & Environment

| Technology | Version | Purpose |
|---|---|---|
| **python-dotenv** | ≥ 1.0.0 | Loads API keys and settings from a `.env` file for local development |

---

## DevOps & Deployment

| Technology | Purpose |
|---|---|
| **Docker** | Containerises the full application (`Dockerfile`) |
| **Docker Compose** | Orchestrates three services: API server, PPO trainer, and self-play trainer |
| **GitHub Actions** | CI/CD pipeline (`.github/` workflows) |

---

## Platform / Standards

| Technology | Version | Purpose |
|---|---|---|
| **OpenEnv** (`openenv-core`) | ≥ 0.2.0 | Standardised RL-environment protocol; `openenv.yaml` describes the env schema |
| **setuptools** | ≥ 68 | Python package build backend (`pyproject.toml`) |
| **uv** | ≥ 0.11.0 | Fast Python package manager / resolver (`uv.lock`) |

---

## Testing

| Technology | Version | Purpose |
|---|---|---|
| **pytest** | ≥ 8.0.0 | Test runner |
| **pytest-asyncio** | ≥ 0.23.0 | Async test support for FastAPI and tool coroutines |

---

## Architecture Summary

```
FORGE/
├── env/            Gymnasium-compatible RL environment + ClaimGraph data structure
├── agents/         LLM (ReAct FSM), Heuristic, PPO, and Adversarial self-play agents
├── tools/          Forensic tool implementations (Wikipedia, Wayback, Wikidata, …)
├── server/         FastAPI OpenEnv REST API
├── training/       PPO training loop
├── scripts/        Helper scripts (self-play, evaluation)
├── app.py          Gradio web UI
└── inference.py    OpenEnv evaluation entry-point
```

### Key Technology Choices

- **Why Gymnasium?** Provides a battle-tested, widely-adopted interface that makes FORGE compatible with any standard RL library (Stable-Baselines3, RLlib, CleanRL, etc.).
- **Why sentence-transformers locally?** Zero-cost, privacy-preserving claim embeddings with no external API dependency — the model runs entirely in-process.
- **Why FastAPI + OpenEnv?** Allows external LLM agents (e.g., GPT-4, Claude) to interact with the environment over HTTP following the OpenEnv standard, without any Python coupling.
- **Why Groq / OpenAI-compatible client?** Groq offers free-tier LLM inference through an API that is 100% compatible with the OpenAI Python client, enabling LLM-agent evaluation at zero cost.
- **Why Docker Compose?** Cleanly separates the API server, PPO training worker, and self-play worker so each can be scaled or disabled independently.
