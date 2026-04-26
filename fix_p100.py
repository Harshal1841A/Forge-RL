import json
import os

NOTEBOOKS = [
    "training/forge_grpo_colab.ipynb",
    "training/red_grpo_colab.ipynb",
    "training/forge_grpo_colab_(1) (1).ipynb"
]

def patch_notebook(filepath):
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 1 (Install bitsandbytes)
    c1 = "".join(nb["cells"][0]["source"])
    if "bitsandbytes" not in c1:
        c1 = c1.replace(
            "!pip install -q trl transformers datasets accelerate peft",
            "!pip install -q trl transformers datasets accelerate peft bitsandbytes"
        )
        nb["cells"][0]["source"] = c1.splitlines(keepends=True)

    # 2. Update Cell 2 (Model Loading: Unsloth fallback for P100)
    c2 = "".join(nb["cells"][1]["source"])
    if "is_p100" not in c2:
        new_c2 = """# Cell 2 — Load model
import torch

_is_p100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7

if _is_p100:
    print("Detected older GPU (Compute < 7.0). Using standard transformers + peft instead of Unsloth.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig

    MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"
    ))
else:
    print("Detected Turing+ GPU. Using Unsloth for optimized training.")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='unsloth/Qwen2.5-3B-Instruct',
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0, bias='none',
        use_gradient_checkpointing='unsloth',
    )
print('Model loaded.')"""
        nb["cells"][1]["source"] = new_c2.splitlines(keepends=True)

    # 3. Update Cell 7 (Evaluation FastLanguageModel logic)
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "FastLanguageModel.for_inference(model)" in src:
            src = src.replace(
                "FastLanguageModel.for_inference(model)",
                "if not _is_p100:\n        FastLanguageModel.for_inference(model)"
            )
            nb["cells"][i]["source"] = src.splitlines(keepends=True)

    # Write changes
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Patched {filepath}")

for nb_path in NOTEBOOKS:
    patch_notebook(nb_path)

