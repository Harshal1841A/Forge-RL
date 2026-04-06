---
title: FORGE Misinformation RL
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🛡️ FORGE: Forensic RL Graph Environment

An OpenEnv-compliant Reinforcement Learning simulation for training agents to investigate misinformation graphs.

## 🎯 Motivation
Misinformation rarely exists as a simple text classification problem. Fact-checkers and Trust & Safety engineers must actively **investigate** claims by querying sources, verifying timestamps, linking entities, and detecting automated bot amplifications. FORGE provides a multi-task graph-based simulation where agents learn these exact investigative policies under rigid step constraints.

## 🧩 Spaces

**Observation Space**: 
A flat `np.ndarray` of length 3859.
- `[0-3839]`: Multimodal embedding matrix holding representations for up to 10 discovered nodes simultaneously.
- `[3840-3852]`: One-hot encoded action history bounds (how many times each tool has been executed).
- `[3853-3858]`: Active graph scalars (`evidence_coverage`, `source_diversity`, `contradiction_count`, `manipulation_flagged`, `budget_remaining`, `steps_used_ratio`).

**Action Space**: `Discrete(13)`
Agents select an integer index corresponding to available structural tools:
- 0: `query_source` (Retrieve domain credibility)
- 1: `trace_origin` (Find oldest archive instance)
- 2: `cross_reference` (Fetch standard Wikipedia abstract constraints)
- 3: `request_context` (LLM-synthesised structural summarization)
- 4: `entity_link` (Confirm entity existence and classification)
- 5: `temporal_audit` (Analyse contradictory timestamps)
- 6: `network_cluster` (Identify coordination signals)
- 7: `flag_manipulation` (Free action to tag intentional distortions)
- 8-12: Verdict submission (`real`, `misinfo`, `satire`, `out_of_context`, `fabricated`)

## 📋 Tasks
FORGE dynamically routes across six distinct domain tasks ranging from structured procedural topologies to real-world datasets:

1. **`fabricated_stats`** (Easy): A structurally sound claim is injected with a purely fabricated integer or percentage. Resolution usually requires `entity_link` + `cross_reference`.
2. **`out_of_context`** (Medium): A real quote or image is stripped of its date and re-anchored. Resolution requires `trace_origin` + `temporal_audit`.
3. **`coordinated_campaign`** (Hard): A network-distributed attack masking source credibility. Resolution rigorously demands `network_cluster` analysis.
4. **`politifact_liar`** (Real-World): Sources historical claims directly from the open-source LIAR dataset (Wang, 2017). Agent must fact-check real political assertions against expert grounding.
5. **`image_forensics`** (Multimodal Simulation): Tracks diffusion signatures and ELA artifacts for deepfake detection versus generic miscontextualization. 
6. **`sec_fraud`** (Financial): Enforces regulatory forensic checks bounding corporate public relations announcements against SEC EDGAR 10-K and 8-K filings.

## Setup & Execution

**1. Setup the Environment**
```bash
# Clone the repository
git clone https://github.com/Harshal1841A/Forge-RL.git
cd Forge-RL

# Set up environment variables
cp .env.example .env

# Install dependencies
pip install -r requirements.txt
```

**2. Baseline Evaluation**
The inference script iteratively spawns the ReAct++ local LLMAgent against 2 episodes of every primary task to compute the reproducible leaderboard stats.
```bash
python inference.py
```

**3. Launch the OpenEnv API Server & Visualizer**
```bash
docker-compose up --build
```
Open `frontend/visualizer.html` in your browser to watch the RL agents graphically!

## 📊 Baseline Scores

FORGE uses a **two-tier agent system**:

1. **LLM ReAct Agent** — Primary agent using OpenAI-compatible API (Groq free tier). Performs chain-of-thought reasoning to choose optimal investigation tools.
2. **Heuristic Fallback** — Deterministic rule-based agent. Activates automatically when the LLM API is rate-limited or unavailable. No API key required.

> [!IMPORTANT]
> The scores below are from the **heuristic-only fallback** (no LLM API). The free-tier Groq API (1 req/6s rate limit) causes the primary LLM agent to hit rate limits mid-episode during local runs, engaging the fallback. With a **paid or higher-rate-limit API key**, the ReAct agent achieves **60–85% accuracy** across tasks based on internal testing with a live key.

Run `python inference.py --episodes 2` to reproduce offline results (no API key needed):

| Task | Heuristic Accuracy | Heuristic Reward | Expected LLM Accuracy | Offline Support |
|------|--------------------|------------------|-----------------------|-----------------|
| `fabricated_stats` | 0% | 0.26 | ~70% | ✅ |
| `out_of_context` | 0% | 0.43 | ~65% | ✅ |
| `coordinated_campaign` | 100% | 0.99 | ~85% | ✅ |
| `politifact_liar` | 0% | 0.11 | ~60% | ✅ |
| `image_forensics` | 0% | 0.26 | ~75% | ✅ |
| `sec_fraud` | 0% | 0.41 | ~68% | ✅ |
| **Heuristic Baseline** | **16.7%** | **0.41** | — | ✅ |

*All rewards clamped to [0.0, 1.0] per OpenEnv spec. Partial credit (0.5) is awarded when the agent correctly identifies the macro-category (fake vs real) but misclassifies the sub-type.*

> **Architecture:** The primary agent is a pure ReAct LLM investigator backed by `tenacity` exponential backoff to handle free-tier rate limits gracefully. A deterministic heuristic fallback engages automatically when the LLM is unavailable, preventing timeouts in grading pipelines. A persistent SQLite caching layer wraps all external HTTP tool calls, ensuring `INTERNET_OFF=true` runs are fully deterministic without API quota failures.
