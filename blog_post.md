# Unmasking the Architecture: How FORGE-RL Models Misinformation as a Sequential Investigation

*By the FORGE-RL Engineering Team — META AI Hackathon 2026*

Misinformation rarely appears as a single false sentence. It spreads as a coordinated sequence of manipulations: source laundering, quote fabrication, context stripping, and amplification across networks. FORGE-RL was built to evaluate that process as a reinforcement learning investigation problem, not a one-shot classification task.

---

## Why Sequential Investigation, Not Classification?

Most fact-check systems return a final label — true or false. That misrepresents how human analysts actually work.

FORGE-RL models the full investigation loop:

- Choose forensic actions (`query_source`, `cross_reference`, `network_cluster`, `temporal_audit`, etc.)
- Gather evidence across multiple steps (budget: 10 actions per episode)
- Submit a verdict — with explicit reward signals tied to accuracy, efficiency, and evidence coverage

The environment is OpenEnv-compatible, exposed through a FastAPI interface, and deployed as a Hugging Face Space combining the backend and the Next.js forensic dashboard in a single Docker container.

---

## Multi-Agent Roles

A single reasoning stream can overfit to its own assumptions. FORGE-RL uses a multi-role analysis pattern where four independent LLMs each hold a distinct investigative perspective:

| Role | Provider | Responsibility |
|---|---|---|
| Forensic Auditor | Groq / Llama-3 70B | Source credibility validation |
| Context Historian | Cerebras / Llama-3.1 70B | Temporal consistency checks |
| Narrative Critic | Mistral / mistral-small | Rhetorical manipulation detection |
| Expert Reviewer | OpenRouter / Llama-3 8B | Final verdict arbitration |

These roles are paired with graph-informed features from the GNN policy (`agents/gnn_policy.py`) and stepwise rewards, so agents are structurally pushed to investigate before deciding.

Using different model providers per role is an intentional design choice: it prevents shared-model bias where a single LLM's internal assumptions dominate the verdict.

---

## A Concrete Scenario: Coordinated Health Misinformation

One registered task (`plandemic`, difficulty: hard) focuses on coordinated health misinformation patterns. The investigation workflow a trained agent learns:

1. `cross_reference` — check claim against trusted health sources
2. `network_cluster` — inspect propagation and bot coordination patterns
3. `flag_manipulation` — evaluate synthetic media signatures
4. `citation_check` — verify cited WHO or institutional sources actually exist
5. `submit_verdict_misinfo` — return verdict with accumulated evidence

The target is not only "is this false?" but "which manipulation primitives were used to construct it?" — the output is a deception chain, not just a label.

---

## Training Evidence

| Artifact | Location |
|---|---|
| Baseline agent metrics | `baselines/results.json` |
| GRPO training notebook | `notebooks/trl_forge_ma.ipynb` |
| Self-play notebook | `notebooks/forge_combined_selfplay (1).ipynb` |
| Training log snapshot | `checkpoints/training_log.json` |
| Reward curve image | `assets/grpo_reward_curve.png` |

To reproduce training results: open `notebooks/trl_forge_ma.ipynb` in Kaggle (30 free GPU hours/week) and run all cells. Expected TED score at step 100: ~0.20 (random baseline: ~0.11).

---

## Links

- [Hugging Face Space](https://huggingface.co/spaces/NeuralHU/forge-rl)
- [GitHub Repository](https://github.com/Godhand-Arnav/Scalar-finals)
- [Main README](../README.md)
- [Submission Runbook](../HACKATHON_README.md)
