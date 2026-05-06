---
title: FORGE-RL
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# FORGE-RL

**Forensic Open Reasoning & Grounding Environment — Reinforcement Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green)](openenv.yaml)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-orange)](https://huggingface.co/spaces/NeuralHU/forge-rl)

> Built during the META AI Hackathon. A reinforcement learning environment for training and evaluating agents that investigate misinformation — not just classify it.

---

## The Problem

In 2018, a two-minute WhatsApp video of a man "kidnapping a child" spread through a village in Jharkhand. It was actually a public awareness clip shot in Pakistan. By the time anyone corrected the record, five people had been lynched.

In 2019, a deepfake of Amit Shah circulated claiming the BJP would scrap OBC reservations. It hit 50,000 WhatsApp groups in 48 hours.

The damage happens the moment a lie *feels* true. Fact-checkers are playing defence — they tell you a claim is false after it has already spread. What we need are detectives: systems that can identify *how* a piece of misinformation was constructed, not just *whether* it is false.

FORGE-RL trains agents to do exactly that. Given a claim, an agent must use forensic investigation tools, gather evidence, and submit a verdict — the same way a human analyst would.

---

## What FORGE-RL Is

FORGE-RL is two things:

1. **An OpenEnv-compatible RL environment** (`openenv.yaml`) — a Gymnasium-style environment where agents investigate misinformation claims step-by-step using a discrete action space of 13 forensic tools.

2. **A multi-agent forensic platform** — a FastAPI backend with a Next.js dashboard where four independent LLMs (Groq, Cerebras, Mistral, OpenRouter) play distinct investigator roles with a GRPO-trained 0.5B Qwen model as the core policy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FORGE-RL System                      │
├───────────────────────────┬─────────────────────────────┤
│   Next.js Frontend        │   FastAPI Backend           │
│   (spatial-saas/)         │   (server/)                 │
│   • Investigation UI      │   • /step  — take action    │
│   • Live reward feed      │   • /grade — score episode  │
│   • Graph visualiser      │   • /episode — history      │
│   • Deepfake viewer       │   • /deepfake — image check │
└───────────────────────────┴──────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │         ForgeEnvironment             │
                    │  (server/forge_environment.py)       │
                    │                                     │
                    │  Obs space: Box[3859] float32        │
                    │  Act space: Discrete[13]             │
                    │  Reward:    [0.001, 0.999]           │
                    └──────────────────┬──────────────────┘
                                       │
        ┌──────────────────────────────▼──────────────────────────────┐
        │                    Multi-Agent Layer (agents/)              │
        │                                                             │
        │  Forensic Auditor   → Groq    / Llama-3 70B               │
        │  Context Historian  → Cerebras/ Llama-3.1 70B             │
        │  Narrative Critic   → Mistral / mistral-small              │
        │  Expert Reviewer    → OpenRouter / Llama-3 8B              │
        │  Blue-Team PPO      → Trained policy (blue_team/)          │
        └─────────────────────────────────────────────────────────────┘
```

---

## Action Space

The agent has 13 discrete actions — 10 investigation tools and 3 terminal verdicts:

| Action | Index | What It Does |
|---|---|---|
| `query_source` | 0 | Check origin credibility of the claim source |
| `cross_reference` | 1 | Compare claim against known fact-checked articles |
| `network_cluster` | 2 | Detect coordinated amplification networks |
| `temporal_audit` | 3 | Check if content has been time-shifted from old events |
| `entity_link` | 4 | Resolve named entities and verify attribution |
| `context_retrieve` | 5 | Fetch surrounding context stripped from the claim |
| `flag_manipulation` | 6 | Detect image/video manipulation signals |
| `image_verify` | 7 | Deep fake detection on media attachments |
| `citation_check` | 8 | Verify cited sources exist and say what is claimed |
| `amplification_scan` | 9 | Count bot-driven spread patterns |
| `submit_verdict_misinfo` | 10 | Terminal — claim is misinformation |
| `submit_verdict_satire` | 11 | Terminal — claim is satire presented as news |
| `submit_verdict_verified` | 12 | Terminal — claim checks out |

Each investigation step costs `−0.01`. A correct early verdict scores up to `0.90 + efficiency_bonus + coverage_bonus`. Exceeding the 10-step budget applies a `−0.50` penalty.

---

## Deception Primitives

FORGE-RL tracks a vocabulary of atomic disinformation tactics — the building blocks that claims are assembled from:

| Primitive | What It Means | Real Example |
|---|---|---|
| `SOURCE_LAUNDER` | Attribute a fake claim to a credible source | "AIIMS doctors confirm..." |
| `TEMPORAL_SHIFT` | Present old events as happening now | 2013 riot footage shared in 2020 |
| `QUOTE_FABRICATE` | Attach fake words to a real person | The 2019 Amit Shah deepfake |
| `CONTEXT_STRIP` | Remove context to make statements misleading | Satire articles shared as real news |
| `CITATION_FORGE` | Invent or distort official sources | "WHO confirms turmeric milk..." |
| `NETWORK_AMPLIFY` | Use bots to manufacture consensus | Coordinated Twitter hashtag trends |
| `SATIRE_REFRAME` | Present satire as factual breaking news | Postcard News-style headlines |
| `ENTITY_SUBSTITUTE` | Swap people or places to change the story | Pakistani flood images as Indian |

---

## Task Suite

Nine tasks are registered in `openenv.yaml` across three difficulty levels:

| Task | Difficulty |
|---|---|
| `fabricated_stats` | easy |
| `verified_fact` | easy |
| `out_of_context` | medium |
| `politifact_liar` | medium |
| `satire_news` | medium |
| `coordinated_campaign` | hard |
| `image_forensics` | hard |
| `sec_fraud` | hard |
| `plandemic` | hard |

---

## GRPO Training (Notebook)

The training pipeline uses GRPO (Group Relative Policy Optimisation) to fine-tune a Qwen 0.5B model on forensic chain prediction. LoRA adapters (r=16) train in FP32 while the base model stays frozen — this resolves FP16 gradient scaling crashes on free GPUs.

```python
# LoRA adapters — solve FP16 gradient crash, cut VRAM in half
model = get_peft_model(model, LoraConfig(
    r=16, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"]
))

# TED: Tactic Edit Distance — Levenshtein distance on the predicted
# deception primitive chain vs. ground truth, normalised to [0, 1].
# Score of 1.0 = perfect chain prediction. Score of 0.0 = entirely wrong.
def reward_fn(completions, prompts=None, true_chains=None, **kwargs):
    return [tactic_edit_distance(extract_chain(c), true_chains[i])
            for i, c in enumerate(completions)]

# GRPO — 4 completions per claim, scored and ranked
config = GRPOConfig(num_generations=4, generation_batch_size=4,
                    max_steps=100, fp16=True)
trainer = GRPOTrainer(model=model, reward_funcs=reward_fn,
                      args=config, train_dataset=dataset)
trainer.train()
```

**Results on a free Kaggle GPU (100 steps, ~5 minutes):**

![Reward curve – GRPO training on Qwen-0.5B](assets/grpo_reward_curve.png)

| Step | TED Score | Notes |
|---|---|---|
| 0 | ~0.11 | Random baseline (untrained) |
| 65 | ~0.13 | Starts beating random |
| 90 | ~0.20 | Final checkpoint |

TED (Tactic Edit Distance) is normalised Levenshtein distance between the model's predicted deception primitive chain and the ground truth chain. **0.0 = entirely wrong prediction. 1.0 = exact match.** Random selection of 8 primitives gives ~0.11. 100 steps on a 0.5B model moves this to 0.20 — a proof-of-concept result, not a production benchmark.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| CUDA GPU (optional) | Any VRAM ≥ 6 GB for training |
| Node.js (dashboard only) | ≥ 18 |

**Free API keys required** (no credit card for any of these):

| Service | Used For | Sign Up |
|---|---|---|
| Groq | Forensic Auditor agent (Llama-3 70B) | [console.groq.com](https://console.groq.com) |
| Cerebras | Context Historian agent (Llama-3.1 70B) | [cloud.cerebras.ai](https://cloud.cerebras.ai) |
| Mistral | Narrative Critic agent | [console.mistral.ai](https://console.mistral.ai) |
| OpenRouter | Expert Reviewer agent | [openrouter.ai](https://openrouter.ai) |
| MediaStack | Live news cross-reference (500/month free) | [mediastack.com](https://mediastack.com) |
| GNews | Source verification (100/day free) | [gnews.io](https://gnews.io) |

---

## Setup

### Option A — Full Platform (Backend + Dashboard)

```bash
git clone https://github.com/Godhand-Arnav/Scalar-finals.git
cd Scalar-finals

# Python environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Open .env and fill in your keys (all free — see Prerequisites above)

# Start the backend
python server/app.py
# → http://localhost:7860

# Start the dashboard (separate terminal)
cd spatial-saas
npm install
npm run dev
# → http://localhost:3000
```

### Option B — RL Environment Only (OpenEnv)

```bash
pip install openenv-core>=0.2.0 gymnasium torch sentence-transformers fastapi uvicorn

# Run against the live HF Space endpoint
python -c "
import openenv
env = openenv.make('NeuralHU/forge-rl')
obs = env.reset()
print(obs)
"
```

### Option C — GRPO Training Notebook

```bash
# Open notebooks/trl_forge_ma.ipynb in Kaggle or Colab
# Kaggle gives 30 free GPU hours/week — no payment needed
# Run all cells top to bottom
# Expected: 'Setup complete.' after cell 1, training starts in cell 3
```

### Option D — Docker

```bash
docker compose up
# Backend → http://localhost:7860
```

---

## Project Structure

```
forge-rl/
├── server/                    # FastAPI backend
│   ├── forge_environment.py   # Core OpenEnv environment
│   ├── main.py                # API routes (/step, /grade, /episode)
│   └── routes/
│       ├── step.py            # Action handler
│       ├── grade.py           # Episode scoring
│       ├── episode.py         # Episode history
│       └── deepfake.py        # Image forensics
├── agents/                    # Agent implementations
│   ├── llm_agent.py           # Multi-provider LLM agent
│   ├── llm_agent_ma.py        # Multi-agent orchestrator
│   ├── blue_ppo_agent.py      # PPO-trained blue team
│   ├── gnn_policy.py          # GNN policy network
│   └── heuristic_agent.py     # Deterministic baseline
├── blue_team/                 # Replay buffer & PPO trainer
├── red_team/                  # Adversarial content generator
├── spatial-saas/              # Next.js investigation dashboard
├── notebooks/                 # GRPO training notebooks
│   └── trl_forge_ma.ipynb     # Main training notebook (Kaggle/Colab)
├── env/                       # Task registry
├── evaluation/                # Eval scripts
├── openenv.yaml               # OpenEnv spec
└── .env.example               # All keys documented and labeled
```

---

## Evaluation

```bash
# Run full OpenEnv validation suite
python run_validations.py

# Run multi-agent evaluation
python run_evaluation_ma.py

# Run unit tests
pytest tests/ -v
```

---

## Why Not Just Use a Classifier?

Classifiers answer "is this true or false?" — the question of a referee. FORGE-RL asks "how was this constructed?" — the question of a detective.

When FORGE-RL analyses a claim, it doesn't just tag it `false`. It outputs a chain: *`SOURCE_LAUNDER → TEMPORAL_SHIFT → NETWORK_AMPLIFY`*. That chain is actionable intelligence. A fact-checking team can use it to find other content with the same fingerprint — the same campaign, run by the same actors, using the same playbook.

---

## Citing

```bibtex
@misc{forge-rl-2026,
  title  = {FORGE-RL: A Reinforcement Learning Environment for Misinformation Forensics},
  author = {FORGE Research Team},
  year   = {2026},
  url    = {https://github.com/Godhand-Arnav/Scalar-finals}
}
```

---

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and deploy.

---

[GitHub](https://github.com/Godhand-Arnav/Scalar-finals) · [HF Space](https://huggingface.co/spaces/NeuralHU/forge-rl) · [OpenEnv Spec](openenv.yaml)
