# FORGE-MA Submission Runbook

This file is a strict evaluator guide for quickly validating the environment against submission requirements.

## 1) Public Entry Point

- Hugging Face Space: [https://huggingface.co/spaces/NeuralHU/forge-rl](https://huggingface.co/spaces/NeuralHU/forge-rl)
- Space type: Docker, combined frontend (`spatial-saas`) + FastAPI backend (`server.main`)

## 2) OpenEnv API

The backend serves OpenEnv-style interaction routes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /actions`
- `GET /tasks`
- `GET /episodes/{id}/grade`

Interactive API docs are available at `/docs` on local backend runs.

## 3) Local Reproduction

```bash
pip install -r requirements.txt
npm --prefix spatial-saas install
python -m uvicorn server.main:app --host 0.0.0.0 --port 7860
```

In a second terminal:

```bash
npm --prefix spatial-saas run dev
```

Then open `http://localhost:3000`.

## 4) RL Training Script / Notebook

- Primary notebook: `training/forge_grpo_colab.ipynb`
- Alternate notebook: `notebooks/trl_forge_ma.ipynb`
- Frameworks used in notebook flow: TRL/GRPO (and Unsloth path for supported GPUs)

## 5) Included Evidence

- Baseline metrics: `baselines/results.json`
- Training log snapshot: `checkpoints/training_log.json`

Current training evidence in-repo is lightweight; for full verification, re-run the notebook and regenerate curves/artifacts in your own runtime.

## 6) Writeup Material

- Technical writeup: `docs/blog_post.md`

No large video binaries are included in-repo.
