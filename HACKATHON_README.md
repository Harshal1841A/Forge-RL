# 🏆 FORGE-MA: Hackathon Submission Guide

Welcome Judges! This document outlines the technical achievements, final repository state, and evaluation instructions for our Meta × HuggingFace OpenEnv Hackathon (Round 2) submission.

## 🌟 The Plandemic Case Study: Our Capstone Benchmark

We have built a custom, rigorous evaluation scenario designed specifically for this presentation: **The Plandemic Benchmark** (`env/tasks/task_plandemic.py`). 

Instead of relying on static strings, this case study utilizes **dynamic LLM procedural generation** to synthesize novel, complex conspiracy theories on the fly. It challenges the Blue Team to unravel sophisticated, coordinated misinformation campaigns, showcasing the system's ability to handle zero-day narrative structures.

---

## 📈 Technical Achievements & Training Metrics

FORGE-MA operates on legitimate, pre-trained neural network checkpoints developed via a 50-generation adversarial curriculum learning pipeline.

### 1. OpenEnv-Compliant RL Environment
- 9 task types, 13-action discrete space, reward range [-1.0, 1.0]
- Full Gymnasium interface: reset() / step() / grade()
- Docker image + HuggingFace Space deployment
- openenv validate passes

### 2. Adversarial Multi-Agent Architecture
- Red Team (HAE: 1-layer MEAN-agg GNN, 32-dim) vs Blue Team (GIN: 2-layer SUM-agg, 64-dim)
- Co-evolutionary self-play: run `python pretrain.py` to generate training checkpoints
- Hierarchical reward shaping (TED × 0.40, F1 × 0.30, Δplausibility × 0.20 + consensus/expert/budget bonuses)
- Anti-reward-hacking: chain entropy bonus + chain length penalty

### 3. Multi-Provider Society of Thought
- 4 agents, 4 different AI companies: Groq (Auditor), Cerebras (Historian), Mistral (Critic), OpenRouter (NegotiatedSearch)
- No shared model bias by design
- GIN topology hint propagated to all LLM agents before verdict

### 4. Production-Grade Server Infrastructure
- FastAPI with circuit breakers, retry-with-jitter, reservoir-sampled latency metrics (p50/p95/p99)
- STIX 2.1 forensic bundle export
- 500-episode in-memory store with LRU eviction

---

## 🛠️ Repository Polish (V4.2)

We have ruthlessly minimized technical debt to provide you with a clean, professional codebase:
* **The Graveyard is Gone:** Removed over 100KB of redundant, legacy code (including `app_head.py`).
* **Transient Purge:** Cleared all temporary caches, leftover training logs, and abandoned scratch scripts.
* **FSM-Compliant Agents:** The `llm_agent.py` logic strictly enforces Finite State Machine actions, completely preventing invalid "verdict jumps" without requisite evidence.
* **UI Modernization:** Replaced the legacy Gradio monolith with a modern, decoupled **Next.js (`frontend/`)** SaaS dashboard backed by a dedicated FastAPI REST API (`server/`).

---

## 🏁 How to Evaluate

1. **Start the API Server (Terminal 1):**
   ```bash
   python -m uvicorn server.main:app --host 0.0.0.0 --port 7860
   ```
2. **Start the Frontend (Terminal 2):**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. **Access the Dashboard:** Open your browser to `http://localhost:3000`.
4. **Run the Capstone:** Select the "Plandemic" scenario from the task dropdown and click "Begin Investigation". Watch as the Multi-Provider Society of Thought dissects the generated claim.
5. **Inspect the Weights:** View the trained checkpoints `checkpoints/gin_model.pt` and `checkpoints/hae_model.pt` that power the backend logic.

*We hope you enjoy reviewing FORGE-MA as much as we enjoyed building it!*
