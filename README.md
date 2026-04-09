---
title: FORGE Misinformation RL
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# FORGE: Forensic RL Graph Environment

> An OpenEnv-compliant reinforcement learning environment for training 
> agents to investigate and classify misinformation — the way human 
> fact-checkers actually do it.

## Why FORGE?

Most misinformation benchmarks treat fact-checking as text classification.
Real Trust & Safety engineers do something fundamentally different: they 
**investigate**. They query source credibility, trace image origins, 
cross-reference with authoritative databases, audit timelines, and map 
coordinated bot amplification networks — all under time pressure.

FORGE simulates exactly this investigative workflow as a sequential 
decision-making problem. An agent must choose which forensic tool to 
apply at each step, accumulate evidence in a claim graph, and submit a 
verdict before its step budget runs out.

**Real-world value:** An agent trained on FORGE could be deployed as an 
automated first-pass content moderation system, triaging claims for human 
review at the speed of social media spread.

## Environment Design

**Observation Space** — `Box(3859,)` float32:
- `[0:3840]` — 384-dim sentence embeddings for up to 10 discovered graph nodes
- `[3840:3853]` — Tool call history (how many times each of 13 actions was used)
- `[3853:3859]` — Graph scalars: `evidence_coverage`, `source_diversity`, 
  `contradiction_count`, `manipulation_flagged`, `budget_remaining`, `steps_used_ratio`

**Action Space** — `Discrete(13)`:

| Index | Action | Description |
|---|---|---|
| 0 | `query_source` | Retrieve domain credibility score |
| 1 | `trace_origin` | Find oldest archival instance |
| 2 | `cross_reference` | Check against Wikipedia/encyclopedic sources |
| 3 | `request_context` | LLM-synthesised structural summary |
| 4 | `entity_link` | Verify entity existence and classification |
| 5 | `temporal_audit` | Analyse timestamp anomalies and EXIF data |
| 6 | `network_cluster` | Detect coordinated bot amplification |
| 7 | `flag_manipulation` | Free action: tag deliberate adversarial intent |
| 8–12 | `submit_verdict_*` | Submit final verdict (real/misinfo/satire/out_of_context/fabricated) |

**Reward Function** — Potential-based dense shaping (Ng et al., 1999):
- Terminal reward: correctness + calibration bonus + efficiency bonus + manipulation component
- Step reward: `Φ(s') − Φ(s)` where `Φ = coverage + diversity + contradiction_area + network_diameter`
- All rewards clipped to open interval `(0.001, 0.999)` per OpenEnv spec

## Tasks

| # | Task | Difficulty | Domain | Key Tools | Grader |
|---|---|---|---|---|---|
| 1 | `fabricated_stats` | Easy | Health/Science | entity_link, cross_reference | ✅ |
| 2 | `verified_fact` | Easy | Control/Negative | cross_reference, entity_link | ✅ |
| 3 | `out_of_context` | Medium | Media | trace_origin, temporal_audit | ✅ |
| 4 | `politifact_liar` | Medium | Politics (LIAR dataset) | cross_reference, entity_link | ✅ |
| 5 | `satire_news` | Medium | Linguistics | request_context, cross_reference | ✅ |
| 6 | `coordinated_campaign` | Hard | Social Networks | network_cluster, query_source | ✅ |
| 7 | `image_forensics` | Hard | Multimodal | temporal_audit, trace_origin | ✅ |
| 8 | `sec_fraud` | Hard | Financial/SEC | cross_reference, entity_link | ✅ |

Each grader awards partial credit for correct tool usage independently 
of verdict correctness, providing dense signal across the full trajectory.

## Setup

**Prerequisites:** Python 3.11+, Docker

```bash
# Clone
git clone https://github.com/Harshal1841A/Forge-RL.git
cd Forge-RL

# Install dependencies
pip install -r requirements.txt

# (Optional) Copy environment config
cp .env.example .env
```

## Running Inference

```bash
# Offline — no API key needed (uses deterministic HeuristicAgent)
python inference.py --episodes 2

# With LLM agent (Groq free tier)
HF_TOKEN=your_groq_key python inference.py --episodes 2

# Full evaluation
python inference.py --episodes 5 --difficulty 2
```

## Docker

```bash
# Build
docker build -t forge-rl .

# Run server
docker run -p 7860:7860 \
  -e HF_TOKEN=your_key \
  -e MODEL_NAME=llama3-8b-8192 \
  forge-rl

# Verify
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "fabricated_stats", "difficulty": 1}'
```

## OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/reset` | Start new episode |
| POST | `/step` | Take action |
| GET | `/state` | Current episode state |
| GET | `/tasks` | List all tasks |
| GET | `/health` | Server health + OpenEnv compliance |
| GET | `/observations/schema` | Typed observation schema |
| GET | `/actions/schema` | Typed action schema |
| GET | `/rewards/schema` | Typed reward schema |

## Baseline Scores

Two-tier agent system: LLM ReAct agent (primary) + deterministic HeuristicAgent 
(fallback when API unavailable — no key required).

| Task | Difficulty | Heuristic | LLM ReAct |
|---|---|---|---|
| `fabricated_stats` | Easy | ~35% | ~70% |
| `verified_fact` | Easy | ~45% | ~80% |
| `out_of_context` | Medium | ~30% | ~65% |
| `politifact_liar` | Medium | ~25% | ~60% |
| `satire_news` | Medium | ~30% | ~65% |
| `coordinated_campaign` | Hard | ~40% | ~85% |
| `image_forensics` | Hard | ~20% | ~75% |
| `sec_fraud` | Hard | ~25% | ~68% |
| **Overall** | | **~31%** | **~71%** |

Scores are reward means across 2 episodes per task. Heuristic scores 
are fully reproducible offline with zero API calls.

## Architecture

```
FORGE
├── env/
│   ├── misinfo_env.py      # Gymnasium-compatible environment
│   ├── claim_graph.py      # Graph data structure for evidence
│   ├── reward.py           # Potential-based reward shaping
│   └── tasks/              # 8 task generators + graders
├── agents/
│   ├── llm_agent.py        # ReAct LLM agent (FSM-constrained)
│   ├── heuristic_agent.py  # Deterministic fallback
│   └── ppo_agent.py        # PPO training agent
├── tools/                  # Simulated forensic tool implementations
├── server/                 # FastAPI OpenEnv server
└── inference.py            # OpenEnv evaluation script
```
