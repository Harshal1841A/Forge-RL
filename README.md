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
python scripts/inference.py
```

**3. Launch the OpenEnv API Server & Visualizer**
```bash
docker-compose up --build
```
Open `frontend/visualizer.html` in your browser to watch the RL agents graphically!

## 📊 Baseline Scores (LLM Hybrid — Groq Free-Tier)

Run `python scripts/inference.py --episodes 2` to reproduce results.

| Task | Accuracy | Mean Reward | Agent | Offline Grading |
|------|----------|-------------|-------|-----------------|
| `fabricated_stats` | 0% | 0.261 | ReAct + Heuristic | ✅ Supported |
| `out_of_context` | 0% | 0.426 | ReAct + Heuristic | ✅ Supported |
| `coordinated_campaign` | 100% | 0.986 | ReAct + Heuristic | ✅ Supported |
| `politifact_liar` | 0% | 0.112 | ReAct + Heuristic | ✅ Supported |
| `image_forensics` | 0% | 0.261 | ReAct + Heuristic | ✅ Supported |
| `sec_fraud` | 0% | 0.410 | ReAct + Heuristic | ✅ Supported |
| **All Tasks (Heuristic Baseline)** | **16.7%** | **0.406** | Heuristic only | ✅ Supported |

*Note: Per openenv validation requirements, all step and baseline rewards are strictly clamped to the [0.0, 1.0] range.*

> ** Architecture:** The primary agent is a pure ReAct LLM investigator backed by `tenacity` exponential backoff to handle Groq free-tier rate limits. A deterministic heuristic fallback engages automatically when the LLM is unavailable or within 2 steps of the budget limit, preventing timeouts in grading pipelines. A persistent SQLite caching layer wraps all external HTTP tool calls, ensuring that with `INTERNET_OFF=true` the environment runs fully deterministically without API quota failures.
