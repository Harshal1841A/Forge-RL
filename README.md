# FORGE v2.1 — Unified Misinformation Forensics Platform

> **FORGE v1** (Gymnasium RL environment) **+** **FORGE-MA v9.0** (Adversarial Multi-Agent RL extension)  
> *Meta × HuggingFace OpenEnv Hackathon — Round 2*

---

## Overview

FORGE v2.1 merges two complementary systems into one codebase and introduces a **multi-provider agent architecture** that eliminates shared-model bias by routing each forensic agent to a different AI provider.

| System | Role |
|---|---|
| **FORGE v1** (`env/misinfo_env.py`) | Production Gymnasium env with 8 task types, tool registry, FastAPI server, GNN policy, PPO training |
| **FORGE-MA v9.0** (`env/forge_env.py`) | Adversarial multi-agent extension: Red Team (HAE) vs Blue Team (SoT + GIN), hierarchical rewards |

---

## 🧠 Multi-Provider Agent Architecture (v2.1)

Each forensic agent uses a **different AI provider** — no shared training data, no shared bias.

| Agent | Provider | Model | Why |
|---|---|---|---|
| **Forensic Auditor** | 🟣 Groq | `llama3-70b-8192` | Strong factual/source reasoning |
| **Context Historian** | 🔵 Cerebras | `llama3.1-70b` | Best at temporal/provenance detection |
| **Narrative Critic** | 🟠 Mistral | `mistral-small-latest` | Excels at narrative style & satire |
| **NegotiatedSearch** | 🟢 OpenRouter | `llama-3-8b:free` | Fast, lightweight tool-selection pre-pass |

A verdict flagged by **all 4 agents** from 4 different companies = extremely high confidence (true cross-model consensus).

> All 4 providers are **100% free** with no credit card required. The system falls back gracefully to deterministic mock logic when any API key is missing.

---

## FORGE v1 Architecture

```
MisInfoForensicsEnv (Gymnasium)
├── 8 Task Types (fabricated_stats, sec_fraud, image_forensics, ...)
├── 13 Actions (tool calls + verdict submission)
├── LLM Agent (OpenAI / Groq)
├── GNN Policy (PPO-trained)
├── Tool Registry (query_source, trace_origin, cross_reference, ...)
└── FastAPI Server (REST API + WebSocket)
```

## FORGE-MA v9.0 Architecture

```
Society of Thought (Blue Team — 4 agents, 4 different AIs)
├── Forensic Auditor   [Groq/Llama]     — leads investigation
├── Context Historian  [Cerebras/Llama]  — temporal framing
├── Narrative Critic   [Mistral]         — quote/satire specialist (P4, P8)
└── Graph Specialist   [BlueGIN local]   — 2-layer GIN (SUM, 64-dim)

Red Team
└── HAE Adversary      — 1-layer GNN (MEAN, 32-dim) + ActionValidator

Hierarchical Reward Shaper
├── TED   × 0.40       — tactic chain edit distance
├── F1    × 0.30       — tactic precision / recall
├── PLB   × 0.20       — plausibility delta
├── Consensus bonus    — ±0.05 / +0.10 unanimous
├── Expert bonus       — +0.05 APPROVE
└── Budget penalty     — −0.01/step, −0.50 over-budget
```

## Key Constraints (FORGE-MA)

| Parameter | Value |
|---|---|
| `K_MAX` (chain length) | **4** |
| Blue GIN layers | **2** (SUM, 64-dim) |
| Red HAE layers | **1** (MEAN, 32-dim) |
| DISARM ID format | **T-prefix only** |
| Plausibility scorer | **Zero LM calls, <1ms** |

---

## Quick Start

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. (Optional) Install FORGE-MA GPU packages
pip install torch-geometric trl stix2

# 3. Copy example env and fill in your free API keys
cp .env.example .env
# Edit .env with your keys (see "Getting Free API Keys" below)

# 4. Run unified Gradio UI (both FORGE v1 + FORGE-MA tabs)
python app.py

# 5. Run FORGE v1 tests
python -m pytest tests/test_graders.py -v

# 6. Run FORGE-MA tests (126 cases)
python -m pytest tests/forge_ma/ -v --tb=short

# 7. Run FORGE-MA evaluation only
python -m evaluation.evaluator

# 8. Start FastAPI server (FORGE v1)
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## 🔑 Getting Free API Keys

All 4 providers are free with no credit card needed:

| Provider | Sign Up | Key format |
|---|---|---|
| **Groq** (Auditor) | [console.groq.com](https://console.groq.com) → API Keys | `gsk_...` |
| **Cerebras** (Historian) | [cloud.cerebras.ai](https://cloud.cerebras.ai) → API Keys | `csk_...` |
| **Mistral** (Critic) | [console.mistral.ai](https://console.mistral.ai) → API Keys | `...` |
| **OpenRouter** (NegotiatedSearch) | [openrouter.ai](https://openrouter.ai) → Keys | `sk-or-...` |

Then in your `.env`:
```env
OPENAI_API_KEY=gsk_...          # Groq key
CEREBRAS_API_KEY=csk_...        # Cerebras key
MISTRAL_API_KEY=...             # Mistral key
OPENROUTER_API_KEY=sk-or-...    # OpenRouter key
```

---

## Project Structure

```
Mis_Information Forensic/
│
├── app.py                      # Unified Gradio UI (FORGE v1 + FORGE-MA tabs)
├── config.py                   # Unified configuration (multi-provider)
├── inference.py                # Model inference utilities
├── pyproject.toml              # Unified dependencies (v2.1.0)
├── requirements.txt            # Full merged requirements
├── .env.example                # Template — copy to .env and fill keys
│
├── env/
│   ├── misinfo_env.py          # FORGE v1: Gymnasium env (8 tasks, 13 actions)
│   ├── claim_graph.py          # FORGE v1: dict-based ClaimGraph
│   ├── reward.py               # FORGE v1: reward functions
│   ├── tasks/                  # 8 misinformation task generators
│   ├── forge_env.py            # FORGE-MA: adversarial Gymnasium env
│   ├── claim_graph_ma.py       # FORGE-MA: dataclass ClaimGraph (ClaimNode/EvidenceEdge)
│   ├── primitives.py           # FORGE-MA: PrimitiveType enum, K_MAX=4
│   ├── episode_output.py       # FORGE-MA: immutable episode result
│   ├── oversight_report.py     # FORGE-MA: markdown report generator
│   └── report_manager.py       # FORGE-MA: Markovian evidence reports
│
├── agents/
│   ├── llm_agent.py            # FORGE v1: LLM agent (OpenAI / Groq)
│   ├── ppo_agent.py            # FORGE v1: PPO agent
│   ├── gnn_policy.py           # FORGE v1: GNN policy
│   ├── heuristic_agent.py      # FORGE v1: rules-based agent
│   ├── llm_agent_ma.py         # FORGE-MA: multi-provider LLM agent (Groq/Cerebras/Mistral/OpenRouter)
│   └── expert_reviewer_agent.py# FORGE-MA: Dawid-Skene / Ising ensemble
│
├── red_team/                   # FORGE-MA
│   ├── hae_model.py            # HAE: 1-layer GNN, MEAN, 32-dim
│   ├── red_agent.py            # Adversarial action proposer
│   └── action_validator.py     # K_MAX + DISARM gates
│
├── blue_team/                  # FORGE-MA
│   ├── gin_predictor.py        # BlueGIN: 2-layer, SUM, 64-dim
│   ├── narrative_critic.py     # Society agent 4 — Mistral (P4/P8)
│   ├── negotiated_search.py    # V_ensemble: Cerebras + Mistral pre-analysis
│   ├── replay_buffer.py        # Episode replay buffer
│   └── society_of_thought.py   # 4-agent orchestration (cross-provider consensus)
│
├── rewards/                    # FORGE-MA
│   ├── hierarchical_reward.py  # Composite R_total shaper
│   ├── tactic_edit_dist.py     # Normalized Levenshtein TED
│   ├── tactic_pr.py            # Precision / Recall / F1
│   ├── plausibility.py         # Deterministic plausibility scorer
│   ├── budget_penalty.py       # Step cost + over-budget shaping
│   └── red_step_reward.py      # Per-step Red agent reward
│
├── evaluation/                 # FORGE-MA
│   └── evaluator.py            # Layer-9 benchmark metrics
│
├── data/
│   └── disarm_registry.json    # FORGE-MA: 8 T-prefix DISARM TTPs
│
├── training/
│   ├── train_ppo.py            # FORGE v1: PPO training loop
│   ├── curriculum.py           # FORGE v1: curriculum scheduler
│   ├── eval.py                 # FORGE v1: evaluation utilities
│   └── ppo_trainer_ma.py       # FORGE-MA: PPO trainer + TrainingStats
│
├── tools/                      # FORGE v1: tool registry
│   ├── tool_registry.py
│   ├── query_source.py
│   ├── trace_origin.py
│   ├── cross_reference.py
│   ├── entity_link.py
│   ├── temporal_audit.py
│   └── network_cluster.py
│
├── server/                     # FORGE v1: FastAPI REST server
│   └── app.py
│
└── tests/
    ├── test_graders.py         # FORGE v1 tests
    └── forge_ma/               # FORGE-MA test suite (126 tests)
        ├── conftest.py
        ├── test_layer3_agents.py
        ├── test_layer4_red_team.py
        ├── test_layer5_rewards.py
        ├── test_layer6_output.py
        ├── test_layer7_training.py
        ├── test_layer9_evaluation.py
        ├── test_plausibility_timing.py
        ├── test_primitives.py
        └── test_tactic_edit_dist.py
```

---

## FORGE-MA Test Coverage

| Layer | Module | Tests | Status |
|---|---|---|---|
| 0 — Foundation | `primitives`, `tactic_edit_dist`, `plausibility` | 33+ | ✅ |
| 1 — Environment | `claim_graph_ma`, `report_manager`, `tactic_pr` | — | ✅ |
| 2 — GIN | `gin_predictor` | — | ✅ |
| 3 — Agents | `llm_agent_ma`, `society_of_thought`, `narrative_critic` | 11 | ✅ |
| 4 — Red Team | `hae_model`, `red_agent`, `action_validator` | 28 | ✅ |
| 5 — Rewards | `hierarchical_reward`, `budget_penalty` | 29 | ✅ |
| 6 — Output | `episode_output`, `oversight_report` | 23 | ✅ |
| 7 — Training | `forge_env`, `ppo_trainer_ma` | 19 | ✅ |
| **Total** | | **126** | **✅** |

---

## License
MIT