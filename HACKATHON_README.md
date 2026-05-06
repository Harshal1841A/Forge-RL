# FORGE-RL Submission Runbook

This file is a strict evaluator guide for quickly validating the environment against submission requirements.

## 1) Public Entry Point

- Hugging Face Space: [https://huggingface.co/spaces/NeuralHU/forge-rl](https://huggingface.co/spaces/NeuralHU/forge-rl)
- Space type: Docker — combined frontend (`spatial-saas`) + FastAPI backend (`server.main`)

## 2) OpenEnv API

The backend serves OpenEnv-compatible interaction routes:

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new investigation episode |
| `/step` | POST | Take a forensic action (0–12) |
| `/state` | GET | Current episode state |
| `/actions` | GET | Available action list |
| `/tasks` | GET | Task registry |
| `/episodes/{id}/grade` | GET | Score a completed episode |

Interactive API docs: `http://localhost:7860/docs` (local) or the Space `/docs` endpoint.

## 3) Local Reproduction

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Configure API keys (all free — see .env.example for signup links)
cp .env.example .env
# Edit .env with your Groq / Cerebras / Mistral / OpenRouter keys

# 3. Start backend
python -m uvicorn server.main:app --host 0.0.0.0 --port 7860

# 4. Start frontend (separate terminal)
npm --prefix spatial-saas install
npm --prefix spatial-saas run dev
```

Open `http://localhost:3000` for the investigation dashboard.

## 4) RL Training Notebooks

| Notebook | Location | Notes |
|---|---|---|
| Primary (GRPO) | `notebooks/trl_forge_ma.ipynb` | Kaggle / Colab compatible |
| Self-play | `notebooks/forge_combined_selfplay (1).ipynb` | Red vs Blue adversarial run |

Frameworks: TRL/GRPO. Unsloth path available for supported GPUs.

## 5) Included Evidence

| File | Contents |
|---|---|
| `baselines/results.json` | Baseline agent metrics |
| `checkpoints/training_log.json` | Training log snapshot |
| `assets/grpo_reward_curve.png` | Reward curve (Qwen 0.5B, 100 steps) |

For full verification, re-run a notebook and regenerate curves in your own runtime.

## 6) Writeup Material

- Technical writeup: `docs/blog_post.md`
- Impact document: `REAL_WORLD_IMPACT.md`

No large video binaries are included in-repo.
