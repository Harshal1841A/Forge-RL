---
title: Forge RL
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# FORGE-MA: Misinformation Forensics RL Environment

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-v14+-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

FORGE-MA is an OpenEnv-compatible environment for investigating misinformation through sequential forensic actions. Instead of only predicting a class label, the environment rewards agents for finding evidence, identifying manipulation tactics, and submitting a final verdict.

## Submission Links

- Hugging Face Space (runnable environment): [https://huggingface.co/spaces/NeuralHU/forge-rl](https://huggingface.co/spaces/NeuralHU/forge-rl)
- Technical writeup: [`docs/blog_post.md`](docs/blog_post.md)
- Judge runbook: [`HACKATHON_README.md`](HACKATHON_README.md)
- Training notebook (Colab-ready): [`training/forge_grpo_colab.ipynb`](training/forge_grpo_colab.ipynb)

## OpenEnv Scope

- 9 tasks covering fabricated stats, out-of-context claims, campaign amplification, satire reframing, image/deepfake workflows, and domain-specific cases.
- Discrete 13-action interface for evidence gathering and verdict submission.
- REST endpoints for reset, step, state, grading, and leaderboard.
- Dockerized Space deployment with combined frontend and backend runtime.

## Quick Start (Local)

```bash
pip install -r requirements.txt
npm --prefix spatial-saas install
python -m uvicorn server.main:app --host 0.0.0.0 --port 7860
```

In a second terminal:

```bash
npm --prefix spatial-saas run dev
```

Open `http://localhost:3000`.

## Training + Evidence

- Main notebook: `training/forge_grpo_colab.ipynb` (TRL/GRPO workflow).
- Baseline outputs: `baselines/results.json`.
- Current training log snapshot: `checkpoints/training_log.json`.

Note: this repository currently includes baseline metrics and a short training-log snapshot. If you are evaluating final score quality, re-run the training notebook for full-length experiments and regenerated curves.

## License

Apache 2.0.
