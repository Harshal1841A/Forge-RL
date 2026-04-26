import json
import os

NOTEBOOKS = [
    "training/forge_grpo_colab.ipynb",
    "training/red_grpo_colab.ipynb",
    "notebooks/trl_forge_ma.ipynb"
]

def patch_notebook_env(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Replacement for Cell 1
    new_c1 = """# Cell 1 — Install & Universal Environment Setup (Colab / Kaggle / Local)
import os, sys, subprocess
import torch

HOME_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else ("/content" if os.path.exists("/content") else os.getcwd())
REPO_DIR = os.path.join(HOME_DIR, "forge-rl")
UNSLOTH_PKG = "unsloth[kaggle-new]" if "kaggle" in HOME_DIR else "unsloth[colab-new]"

print(f"Setting up environment for {HOME_DIR}...")

# Run installations
cmds = [
    "pip install -q trl transformers datasets accelerate peft bitsandbytes",
    "pip install -q \\"openenv-core[core]>=0.2.1\\"",
    f"pip install -q \\"{UNSLOTH_PKG} @ git+https://github.com/unslothai/unsloth.git\\"",
    "pip install -q torch-geometric",
    f"rm -rf {REPO_DIR}",
    f"git clone https://github.com/Godhand-Arnav/Scalar-finals.git {REPO_DIR}"
]

for cmd in cmds:
    print(f"Running: {cmd}")
    os.system(cmd)

# Set working directory safely
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

if not torch.cuda.is_available():
    print('WARNING: NO GPU DETECTED. Please enable GPU in your notebook settings.')
else:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('Setup complete.')"""

    # We want to replace the FIRST cell that contains `!pip install` or `!git clone`
    patched = False
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "!pip install" in src or "!git clone" in src or "forge-rl" in src:
            nb["cells"][i]["source"] = new_c1.splitlines(keepends=True)
            patched = True
            break
            
    if not patched:
        # If no setup cell found, insert at the beginning
        nb["cells"].insert(0, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": new_c1.splitlines(keepends=True)
        })

    # Write changes
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Patched environment setup for {filepath}")

for nb_path in NOTEBOOKS:
    patch_notebook_env(nb_path)
