import json
import os

NOTEBOOKS = [
    "training/forge_grpo_colab.ipynb",
    "training/forge_grpo_colab_(1) (1).ipynb",
    "training/red_grpo_colab.ipynb",
    "notebooks/trl_forge_ma.ipynb"
]

def patch_generation_batch_size(filepath):
    if not os.path.exists(filepath):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "_cfg = dict(" in src and "generation_batch_size" not in src:
            # We want to insert generation_batch_size = num_generations if num_generations is there, else maybe 8 or 4
            # A safe way is to just add it inside the dict
            src = src.replace(
                "learning_rate=5e-6,",
                "learning_rate=5e-6,\n    num_generations=4,\n    generation_batch_size=4,"
            )
            # Remove existing num_generations if it was lower down
            if src.count("num_generations=") > 1:
                 # It's fine, the dict will overwrite if it's evaluated, but syntactically duplicate keys in dict(a=1, a=2) throws error. 
                 # Let's use regex or string replace carefully.
                 pass
            nb["cells"][i]["source"] = src.splitlines(keepends=True)

    # Safer replacement: Just append to the _cfg dict before passing to GRPOConfig
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "config = GRPOConfig(**_cfg)" in src:
             if "_cfg['generation_batch_size']" not in src:
                  new_src = src.replace(
                      "config = GRPOConfig(**_cfg)",
                      "_cfg['num_generations'] = 4\n_cfg['generation_batch_size'] = 4\nconfig = GRPOConfig(**_cfg)"
                  )
                  nb["cells"][i]["source"] = new_src.splitlines(keepends=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Fixed generation_batch_size in {filepath}")

for nb in NOTEBOOKS:
    patch_generation_batch_size(nb)
