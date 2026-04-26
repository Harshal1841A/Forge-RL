import json

NB = "training/forge_grpo_colab.ipynb"
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

c1 = "".join(nb["cells"][0]["source"])
c1 = c1.replace(
    "!git clone https://huggingface.co/spaces/NeuralHU/forge-rl /content/forge-rl",
    "!git clone https://github.com/Godhand-Arnav/Scalar-finals.git /content/forge-rl"
)
nb["cells"][0]["source"] = c1.splitlines(keepends=True)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Done - Cell 1 now clones from GitHub")
