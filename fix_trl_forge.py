import json

NB = "notebooks/trl_forge_ma.ipynb"
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- FIX CELL 4 (GRPOConfig & Trainer) ---
c4 = "".join(nb["cells"][3]["source"])

old_config = """config = GRPOConfig(
    output_dir="./forge_ma_grpo",
    num_train_epochs=1,
    max_steps=100,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    max_new_tokens=64,
    temperature=0.7,
    logging_steps=10,
    save_steps=50,
    report_to="none",
)"""

new_config = """# Version-safe config builder — works on all TRL versions
import inspect
_valid = set(inspect.signature(GRPOConfig).parameters.keys())
_cfg = dict(
    output_dir="./forge_ma_grpo",
    num_train_epochs=1,
    max_steps=100,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=50,
    report_to="none",
    fp16=True,  # Usually needed for GPU training
)
# max_completion_length (TRL 1.x) or max_new_tokens (TRL 0.x)
for k, v in [('max_completion_length', 64), ('max_new_tokens', 64)]:
    if k in _valid:
        _cfg[k] = v
        break

config = GRPOConfig(**_cfg)"""

c4 = c4.replace(old_config, new_config)

# Also fix the trainer call to handle processing_class / tokenizer differences
old_trainer = """trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=dataset,
    reward_funcs=reward_wrapper,
)"""

new_trainer = """try:
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_wrapper,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
except TypeError:
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_wrapper,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )"""

c4 = c4.replace(old_trainer, new_trainer)
nb["cells"][3]["source"] = c4.splitlines(keepends=True)

# --- FIX CELL 5 (Reward plotting) ---
c5 = "".join(nb["cells"][4]["source"])

old_plot = """history = trainer.state.log_history
steps = [h["step"] for h in history if "train_reward" in h]
rewards = [h["train_reward"] for h in history if "train_reward" in h]

if not steps:
    steps = [h["step"] for h in history if "reward" in h]
    rewards = [h.get("reward", h.get("train_reward", 0)) for h in history if "reward" in h or "train_reward" in h]"""

new_plot = """# Version-safe log extraction
steps, rewards = [], []
REWARD_LOG_KEYS = (
    'rewards/mean', 'reward/mean', 'reward',
    'train/reward', 'rewards/reward_mean', 'mean_reward', 'train_reward'
)

if 'trainer' in locals():
    history = trainer.state.log_history
    for h in history:
        if 'step' not in h: continue
        for key in REWARD_LOG_KEYS:
            if key in h:
                steps.append(h['step'])
                rewards.append(h[key])
                break
else:
    print("Trainer not found! Ensure training completed successfully.")
    steps, rewards = [0], [0]"""

c5 = c5.replace(old_plot, new_plot)
nb["cells"][4]["source"] = c5.splitlines(keepends=True)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("Done fixing notebooks/trl_forge_ma.ipynb")
