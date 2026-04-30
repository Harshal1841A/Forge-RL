# Unmasking the Truth: How FORGE-RL Tackles Coordinated Misinformation

*By the FORGE-MA Engineering Team*

Misinformation rarely appears as a single false sentence. It spreads as a coordinated sequence of manipulations: source laundering, quote fabrication, context stripping, and amplification. FORGE-MA was built to evaluate that process as an RL investigation problem instead of a one-shot classification task.

## Why We Built FORGE-MA

Most fact-check systems return only a final label. FORGE-MA models the full investigation loop:

- choose forensic actions (`query_source`, `trace_origin`, `network_cluster`, etc.),
- gather evidence over multiple steps,
- submit a verdict with explicit reasoning signals.

The environment is OpenEnv-compatible and exposed through a FastAPI interface, with a Space deployment that combines backend and frontend in one runnable app.

## Society of Thought + Graph Signals

A single reasoning stream can overfit to its own assumptions. FORGE-MA uses a multi-role analysis pattern:

- Forensic auditor for source validation,
- Context historian for temporal consistency checks,
- Narrative critic for rhetorical manipulation,
- Graph-specialist path for relational claim/evidence structure.

These roles are paired with graph-informed features and stepwise rewards so agents are pushed to investigate before deciding.

## Plandemic-Style Campaign Scenario

One demonstration scenario focuses on coordinated health misinformation patterns. The workflow is:

1. cross-reference claims against trusted sources,
2. inspect propagation and clustering behavior,
3. evaluate manipulation signatures,
4. submit a final verdict.

The target is not only "is this false?" but "which manipulation primitives were used?".

## Training and Current Evidence

The repository includes:

- baseline metrics in `baselines/results.json`,
- GRPO/TRL training notebooks in `training/forge_grpo_colab.ipynb` and `notebooks/forge_combined_selfplay (1).ipynb`,
- a training-log snapshot in `checkpoints/training_log.json`.

This provides an auditable starting point for judges to rerun experiments with their own compute/runtime settings.

## Links

- Hugging Face Space: https://huggingface.co/spaces/NeuralHU/forge-rl
- Main README: `README.md`
- Submission runbook: `HACKATHON_README.md`
- Technical writeup: `docs/blog_post.md`
