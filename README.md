<div align="center">

# 🛡️ FORGE: Unified Misinformation Forensics Platform
**v2.1** — *Adversarial Multi-Agent Reinforcement Learning Edition*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Meta × HuggingFace](https://img.shields.io/badge/Hackathon-OpenEnv_Round_2-blueviolet.svg)](#)

*Fact-checking isn't just classification. It's an investigation.*

</div>

---

## 📖 Overview

**FORGE** (Forensic Operation & Research Graph Environment) bridges the gap between simple true/false classification and authentic human investigative processes. By modeling fact-checking as a sequential investigation across a dynamic toolset, FORGE sets a new standard for AI-driven forensics.

**FORGE v2.1** merges our foundational Gymnasium environment with our state-of-the-art **Multi-Agent Society of Thought**, introducing a revolutionary **Multi-Provider Agent Architecture** that eliminates shared-model bias and introduces adversarial curriculum learning.

| Core System | Functionality |
|---|---|
| **FORGE The Engine**<br>`env/misinfo_env.py` | Production-grade Gymnasium environment supporting 8 forensic task types, a comprehensive tool registry, a Graph Neural Network (GNN) policy, and PPO training. |
| **FORGE-MA The Brain**<br>`env/forge_env.py` | Adversarial multi-agent extension featuring a Red Team (Hierarchical Adversarial Encoder) vs. a Blue Team (Society of Thought + Graph Isomorphism Network), optimized via hierarchical reward shaping. |

---

## 🧠 The Multi-Provider Moat

To prevent homogeneous reasoning loops and shared training data biases, FORGE-MA implements a **zero-shared-bias architecture**. Each specialized agent role is powered by a different frontier AI provider.

| Forensic Role | AI Provider | Selected Model | Strategic Advantage |
|---|---|---|---|
| **Forensic Auditor** | 🟣 Groq | `llama3-70b-8192` | Unparalleled factual and source reasoning. |
| **Context Historian** | 🔵 Cerebras | `llama3.1-70b` | High-speed temporal and provenance detection. |
| **Narrative Critic** | 🟠 Mistral | `mistral-small-latest` | Exceptional at decoding narrative style and satire. |
| **NegotiatedSearch** | 🟢 OpenRouter | `llama-3-8b:free` | Ultra-fast tool-selection and routing pre-pass. |

> **Unanimous Consensus:** A verdict flagged by all four distinct providers represents a rigorously cross-validated, high-confidence conclusion.
> 
> *Note: All configured providers offer generous free tiers. The system gracefully degrades to deterministic mock fallbacks if API keys are missing, ensuring uninterrupted execution.*

---

## 🏗️ System Architecture

### The Blue Team: Society of Thought
Our defensive mechanism relies on a consensus-driven approach, combining large language models with specialized graph-based reasoning.

*   **Forensic Auditor [Groq]:** Leads the core investigation.
*   **Context Historian [Cerebras]:** Analyzes the temporal framing of claims.
*   **Narrative Critic [Mistral]:** Evaluates the stylistic and rhetorical features.
*   **Graph Specialist [BlueGIN]:** A 2-layer Graph Isomorphism Network (SUM pooling, 64-dim) that processes the evolving evidence topology.

### The Red Team: Adversarial Generation
*   **HAE Adversary:** A Hierarchical Adversarial Encoder (MEAN pooling, 32-dim) paired with an Action Validator, designed to intelligently mutate claims and evade detection.

### Hierarchical Reward Shaper
Our sophisticated reward function incentivizes precise and efficient investigations:
*   `TED × 0.40`: Tactic chain edit distance (closeness to truth).
*   `F1 × 0.30`: Tactic precision/recall.
*   `PLB × 0.20`: Plausibility delta.
*   `Consensus & Expert Bonuses`: Rewards for unanimous cross-model agreement.
*   `Budget Penalty`: Enforces efficiency ($-0.01$/step, heavily penalized if over-budget).

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Harshal1841A/Forge-RL.git
cd Forge-RL
```

### 2. Installation

Install the core dependencies:
```bash
pip install -r requirements.txt
```

*(Optional)* Install FORGE-MA GPU packages for advanced training:
```bash
pip install torch-geometric trl stix2
```

### 2. Configuration
Copy the environment template and configure your free API keys:
```bash
cp .env.example .env
# Edit .env with your provider keys
```

### 3. Launch the Dashboard
Start the unified Gradio UI (access both FORGE v1 and FORGE-MA interfaces):
```bash
python app.py
```
*The dashboard will be available at `http://localhost:7860`.*

### 4. Testing
Run the comprehensive test suites to verify integrity:
```bash
# Run core FORGE v1 tests
python -m pytest tests/test_graders.py -v

# Run the full FORGE-MA adversarial test suite (126 cases)
python -m pytest tests/forge_ma -v
```

---

<div align="center">
<i>Built with 🛡️ for a more truthful web.</i>
</div>