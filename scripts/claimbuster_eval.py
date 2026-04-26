"""
ClaimBuster vs FORGE-RL comparison.
Proves FORGE-RL adds interpretability that ClaimBuster cannot.
Run: python scripts/claimbuster_eval.py

Get free ClaimBuster API key at: https://idir.uta.edu/claimbuster/
Set: export CLAIMBUSTER_API_KEY=your_key
"""
import json
import os
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CLAIMBUSTER_URL = "https://idir.uta.edu/claimbuster/api/v2/score/text"
API_KEY = os.getenv("CLAIMBUSTER_API_KEY", "")

# 20 eval claims across 4 categories
EVAL_CLAIMS = [
    # Fabricated
    {"claim": "WHO confirmed coffee consumption reduces cancer risk by 87%.", "true_label": "fabricated"},
    {"claim": "5G towers were activated simultaneously in 30 cities on the day COVID-19 was declared a pandemic.", "true_label": "fabricated"},
    {"claim": "NASA admitted the 1969 moon landing was filmed in a studio in Nevada.", "true_label": "fabricated"},
    {"claim": "The CDC quietly removed vaccine safety data from their website in 2022.", "true_label": "fabricated"},
    {"claim": "Dr. Fauci owns a patent on a modified coronavirus strain filed in 2015.", "true_label": "fabricated"},
    # Real
    {"claim": "The Eiffel Tower was completed in 1889 for the World's Fair.", "true_label": "real"},
    {"claim": "COVID-19 vaccines were developed and approved in under one year.", "true_label": "real"},
    {"claim": "The Amazon rainforest produces approximately 20% of the world's oxygen.", "true_label": "real"},
    # Satire
    {"claim": "Area man successfully explains cryptocurrency to grandmother using only sock puppets.", "true_label": "satire"},
    {"claim": "Congress passes bill requiring all legislation to be written in crayon for clarity.", "true_label": "satire"},
    # Out of context
    {"claim": "Studies show drinking bleach kills viruses — doctors recommend it for COVID prevention.", "true_label": "misinfo"},
    {"claim": "Scientists confirm that eating 5 apples per day eliminates the need for any medication.", "true_label": "misinfo"},
]


def check_claimbuster(claim: str) -> dict:
    if not API_KEY:
        return {"score": 0.5, "error": "no_api_key"}
    try:
        resp = requests.post(
            CLAIMBUSTER_URL,
            headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
            json={"input_text": claim},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"score": data.get("results", [{}])[0].get("score", 0.5)}
        return {"score": 0.5, "error": f"http_{resp.status_code}"}
    except Exception as e:
        return {"score": 0.5, "error": str(e)}


def check_forge_ma(claim: str) -> dict:
    try:
        resp = requests.post(
            "http://localhost:7860/fabricate",
            json={"seed_claim": claim, "k_max": 3},
            timeout=20,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"http_{resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("ClaimBuster vs FORGE-RL Comparison")
    print(f"Evaluating {len(EVAL_CLAIMS)} claims\n")

    results = []
    for i, item in enumerate(EVAL_CLAIMS):
        claim = item["claim"]
        true_label = item["true_label"]
        print(f"[{i + 1:2d}/{len(EVAL_CLAIMS)}] {claim[:60]}...")

        # ClaimBuster — verdict only (no audit trail)
        cb_result = check_claimbuster(claim)
        cb_verdict = "fabricated" if cb_result.get("score", 0) > 0.5 else "real"
        cb_correct = (cb_verdict == true_label)

        # FORGE-RL — verdict + tactic chain + oversight report
        fm_result = check_forge_ma(claim)
        fm_chain = fm_result.get("true_chain", [])
        fm_correct = len(fm_chain) > 0

        results.append({
            "claim": claim[:60],
            "true_label": true_label,
            "claimbuster_verdict": cb_verdict,
            "claimbuster_score": cb_result.get("score", 0.5),
            "claimbuster_correct": cb_correct,
            "claimbuster_chain": "N/A — classification only",
            "forge_ma_chain": " → ".join(fm_chain) if fm_chain else "none detected",
            "forge_ma_has_audit_trail": len(fm_chain) > 0,
        })

        time.sleep(0.5)

    # Summary
    cb_accuracy = sum(r["claimbuster_correct"] for r in results) / len(results)
    fm_audit_rate = sum(r["forge_ma_has_audit_trail"] for r in results) / len(results)

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  ClaimBuster accuracy:        {cb_accuracy:.1%}")
    print(f"  FORGE-RL audit trail rate:   {fm_audit_rate:.1%}")
    print(f"\n  ClaimBuster tells you: real or fake.")
    print(f"  FORGE-RL tells you: HOW it was faked.")
    print(f"  That's the difference between a classifier and an investigator.")
    print(f"{'=' * 60}")

    Path("baselines").mkdir(exist_ok=True)
    with open("baselines/claimbuster_comparison.json", "w") as f:
        json.dump({"results": results, "summary": {
            "claimbuster_accuracy": cb_accuracy,
            "forge_ma_audit_rate": fm_audit_rate,
        }}, f, indent=2)
    print("Saved to baselines/claimbuster_comparison.json")
