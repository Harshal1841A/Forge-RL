---
title: "FORGE-MA: Forensic Multi-Agent Misinformation Detection"
emoji: "🔍"
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
---

# FORGE-MA: Forensic Multi-Agent Investigation of Misinformation

**FORGE-MA** is the first open-source, multi-agent adversarial reinforcement learning environment for *forensic-grade* misinformation detection.

Unlike classifiers that output a single label, FORGE-MA **investigates**: it reconstructs the *tactic chain* — the sequence of deception primitives used to fabricate a claim.

## The Problem

| System | Output | Interpretability |
|--------|--------|-----------------|
| ClaimBuster | `{score: 0.72}` | None |
| Full Fact | `"Mostly False"` | Editorial |
| **FORGE-MA** | `SOURCE_LAUNDER → QUOTE_FABRICATE → TEMPORAL_SHIFT` | **Forensic audit trail** |

FORGE-MA doesn't just tell you a claim is fake — it tells you **how** it was faked.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Red Team   │───→│  Claim Graph │←───│  Blue Team   │
│  (HAE+GIN)  │    │   (STIX-2)   │    │ (SoT + GIN)  │
└─────────────┘    └──────────────┘    └──────────────┘
       ↓                  ↓                    ↓
  Fabrication        Evidence Graph       Investigation
  Primitives         Construction         & Verdict
```

### 8 Deception Primitives (DISARM-Aligned)

| Primitive | DISARM ID | Description |
|-----------|-----------|-------------|
| SOURCE_LAUNDER | T0013.001 | Insert low-trust intermediary domain |
| TEMPORAL_SHIFT | T0046 | Backdate publication timestamp |
| ENTITY_SUBSTITUTE | T0075.001 | Replace named entities |
| QUOTE_FABRICATE | T0006 | Fabricate attributed quotes |
| CONTEXT_STRIP | T0019.001 | Remove qualifying context |
| CITATION_FORGE | T0016 | Create fake academic citations |
| NETWORK_AMPLIFY | T0049 | Simulate coordinated amplification |
| SATIRE_REFRAME | T0085.001 | Repackage satire as news |

## Results

### Measured Baselines (from `baselines/results.json`)

| Agent | TED Score | Accuracy | Status |
|-------|-----------|----------|--------|
| Random | 0.11 | — | Measured |
| v0 (Heuristic) | 0.58 (mean) | 82% | Measured |
| v1 (Prompted LLM) | 0.76 (mean) | 94% | Measured |

### Key Innovation: Tactic Edit Distance (TED)

TED is a **position-weighted** metric that rewards partial chain matches:

```
TED(predicted, true) = Σᵢ wᵢ · 𝟙[pᵢ = tᵢ]
where wᵢ = 1 - (i / max(|p|, |t|))
```

This means getting the *first* primitive right matters more than the last — matching how forensic analysts prioritize root cause identification.

## Live Demo

The deployed Space and local Spatial SaaS dashboard provide:

1. **Live Claim Input** — Type any claim, Red Team fabricates it, Blue Team investigates
2. **GNN Explainer** — Visualize which evidence nodes influenced the chain prediction
3. **Society of Thought** — Watch 4 specialist agents debate the verdict
4. **Oversight Report** — Full forensic audit trail with STIX-2 export

## Training

Fine-tune with GRPO using the provided Colab notebook:

```bash
# Open training/forge_grpo_colab.ipynb in Google Colab
# Uses TRL GRPO with FORGE reward function
# Baselines and training logs are tracked in-repo
```

## License

Apache 2.0
