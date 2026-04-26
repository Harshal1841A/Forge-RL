import json

NB = "training/forge_grpo_colab.ipynb"
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

c6 = "".join(nb["cells"][5]["source"])

old = """for k, v in [('max_completion_length', 128), ('max_new_tokens', 128),
             ('temperature', 0.7), ('generate_kwargs', {'temperature': 0.7, 'do_sample': True})]:
    if k in _valid:
        _cfg[k] = v
config = GRPOConfig(**_cfg)
print(f'GRPOConfig created ({len(_cfg)} params)')"""

new = """# max_completion_length (TRL 1.x) or max_new_tokens (TRL 0.x)
for k, v in [('max_completion_length', 128), ('max_new_tokens', 128)]:
    if k in _valid:
        _cfg[k] = v
        break  # only set one
# temperature is NOT set in GRPOConfig — unsloth model.generate() handles it in Cell 7
config = GRPOConfig(**_cfg)
print(f'GRPOConfig created ({len(_cfg)} params)')"""

c6 = c6.replace(old, new)
nb["cells"][5]["source"] = c6.splitlines(keepends=True)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("Done - temperature removed from GRPOConfig")
