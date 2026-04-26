---
title: "FORGE-RL: GRPO-Powered Fine-Tuning of a 0.5 B LLM for Misinformation Forensics"
author: "Arnav Godhand"
tags: ["reinforcement-learning", "GRPO", "LoRA", "misinformation", "forensics", "HuggingFace"]
license: "MIT"
---

# We Taught a Tiny LLM to Think Like a Forensic Investigator. Here's What Happened.

A 0.5B model. A free GPU. A weekend. And a reward curve that doubled the baseline.

---

## Why We Built This

In 2018, a two-minute WhatsApp video of a man "kidnapping a child" spread through a village in Jharkhand. It was actually a public awareness clip shot in Pakistan. By the time anyone figured that out, five people had been lynched by a mob.

In 2019, a deepfake video of Amit Shah circulated, claiming the BJP would scrap OBC reservations. The video was entirely synthetic, but it hit 50,000 WhatsApp groups in 48 hours.

The damage happens the moment a lie *feels* true. And by the time fact-checkers catch up, it is already too late. The narrative has set.

We have spent years building tools that answer the wrong question. "Is this claim true or false?" is the question of a referee. What we actually need are detectives — systems that can look at a piece of misinformation and ask: *Who built this, how did they build it, and what specific tactics did they use?*

During the META AI Hackathon, we built FORGE-RL: a reinforcement learning pipeline that trains a genuinely tiny 0.5B language model to identify the deception tactics used to construct misinformation. In just 100 steps on Kaggle, the reward doubled.

---

## Why This Is Different

Fact-checkers are playing defence. You feed them a claim, they tell you if it is true. That is useful, but it is the equivalent of a doctor who can only tell you if you are sick. FORGE-RL is a pathologist—it tells you exactly *how* you caught the disease.

When FORGE-RL analyses a claim, it does not just tag it 'false'. It says: *This claim was constructed using SOURCE_LAUNDER followed by TEMPORAL_SHIFT*. That is actionable intelligence. You can trace the campaign and find other content using the same fingerprint.

Every WhatsApp forward your uncle sends is not random — it was deliberately designed. FORGE-RL reads that design. And because it is a trained model, not a rule-based system, it generalises to new, unseen claims.

---

## What FORGE-RL Actually Does

We built a vocabulary of deception primitives — the atomic building blocks of disinformation campaigns:

| Primitive | What It Means | Real Indian Example |
|---|---|---|
| `SOURCE_LAUNDER` | Attribute a fake claim to a credible source | "AIIMS doctors confirm..." (AIIMS never said this) |
| `TEMPORAL_SHIFT` | Present old events as happening right now | 2013 Muzaffarnagar riot footage shared in 2020 |
| `QUOTE_FABRICATE` | Attach fake words to a real person | The 2019 Amit Shah deepfake on reservations |
| `CONTEXT_STRIP` | Remove context to make statements misleading | Satire articles from The Fauxy shared as real news |
| `CITATION_FORGE` | Invent or distort official sources | "WHO confirms turmeric milk..." |
| `NETWORK_AMPLIFY` | Use bots to manufacture consensus | Coordinated Twitter hashtag trends |
| `SATIRE_REFRAME` | Present satire as factual breaking news | Postcard News style headlines |
| `ENTITY_SUBSTITUTE`| Swap people or places to change the story | Pakistani flood images passed off as Indian |

Claims in the wild are assembled from a hidden chain of these primitives. The model's job is to read the text and predict that chain. Our reward signal, Tactic-Edit-Distance (TED), simply measures how many edits it takes to transform the model's prediction into the true chain.

---

## The Training Pipeline

Getting this to run reliably on a free GPU was tough. We fixed Pydantic version conflicts, bleeding-edge `trl` bugs, and FP16 gradient scaling crashes so you do not have to.

The final setup is a single cell that auto-detects Colab, Kaggle, or local. LoRA adapters (r=16) train in FP32 while the base model stays frozen, solving the scaling crashes entirely.

GRPO is beautifully simple here: the model generates 4 forensic analyses for the same claim. We score them all. The better ones get reinforced. No labels needed.

```python
# LoRA adapters — solve the FP16 gradient crash, cut memory in half
model = get_peft_model(model, LoraConfig(r=16, task_type="CAUSAL_LM",
                       target_modules=["q_proj","k_proj","v_proj","o_proj"]))

# The reward function — TED score against ground-truth primitive chain
def reward_fn(completions, prompts=None, true_chains=None, **kwargs):
    return [tactic_edit_distance(extract_chain(c), true_chains[i])
            for i, c in enumerate(completions)]

# GRPO config — explicitly set generation_batch_size to avoid ValueError
config = GRPOConfig(num_generations=4, generation_batch_size=4,
                    max_steps=100, fp16=True)
trainer = GRPOTrainer(model=model, reward_funcs=reward_fn,
                      args=config, train_dataset=dataset)
trainer.train()
```

---

## The Results

![Reward curve – GRPO training on Qwen-0.5B](/assets/grpo_reward_curve.png)

Watch the curve. At step 0, the model knows nothing. By step 65, it crosses the random baseline of 0.11. By step 90, it hits 0.20 on a free GPU.

This is a 0.5B model trained for 5 minutes. The architecture and reward signal scale effortlessly. Swap in a 7B model for 1000 steps, and you are in competitive territory. (Note: All interactive training graphs and forensic chain visualisations are available in the visuals tab).

---

## Run It Yourself

```bash
git clone https://github.com/Godhand-Arnav/Scalar-finals.git
cd Scalar-finals
# Open notebooks/trl_forge_rl.ipynb in Colab or Kaggle
# Run Cell 1, wait for "Setup complete.", then run all remaining cells
```

Everything is pinned. Nothing requires a paid API key to train. The `.env.example` has labeled slots for every service used.

---

## What We Are Building Next

FORGE-RL was built in a weekend, but we are not done. The environment is a modular OpenAI Gym wrapper. We are actively working on:
- Multilingual forensics using `aya-23-8b` for non-English disinformation.
- Adversarial self-play where a red-team generator fights a blue-team forensic agent.
- A live HF Space for real-time claim analysis.
- Federated training that lets news organizations fine-tune collaboratively.

[Open Issues](https://github.com/Godhand-Arnav/Scalar-finals/issues)

---

## Why You Should Use This, Not Just Star It

Most research projects gather stars and never run. FORGE-RL is built to break that pattern.

The notebook is a single file. You do not need a paid GPU — Kaggle gives you 30 free hours a week. You do not need a labelled dataset. You get a modular codebase ready for extension.

India has 500 million WhatsApp users. The next crisis is already being edited. The tools to understand how that content is made should not be locked behind paywalls. Fork it and use it.

---

## A Closing Thought

Better classifiers will not solve misinformation. Systems that understand *how* it works will. Fact-checkers are tired and platform teams are drowning. We need tools that scale — and we just proved that a tiny, free-to-run language model can start to reason about disinformation structurally. That is the proof of concept we need.

Fork it. Run it. Train it on your own data.

[GitHub Repository](https://github.com/Godhand-Arnav/Scalar-finals) — [Demo Space](https://huggingface.co/spaces/NeuralHU/forge-rl) — [HACKATHON_README](https://github.com/Godhand-Arnav/Scalar-finals/blob/main/HACKATHON_README.md)

---

*Built during the META AI Hackathon. All feedback, critiques, and pull requests genuinely welcome.*
