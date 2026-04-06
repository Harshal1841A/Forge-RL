"""
download_liar.py — Download the open-source LIAR dataset (Politifact)

The LIAR dataset contains ~12,836 real political claim statements from PolitiFact.com
along with their ground-truth labels. It is completely open-source (Wang, 2017).

Dataset source: https://huggingface.co/datasets/liar
Direct TSV mirrors: https://github.com/liyuanlucasliu/LIAR

Usage:
    python scripts/download_liar.py
    python scripts/download_liar.py --output data/liar_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── LIAR dataset column headers ─────────────────────────────────────────────
# Columns per the original Wang (2017) paper:
# 0: ID  1: label  2: statement  3: subject  4: speaker  5: speaker_job
# 6: state_info  7: party  8: barely_true_count  9: false_count
# 10: half_true_count  11: mostly_true_count  12: pants_on_fire_count  13: context
COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "speaker_job", "state_info", "party",
    "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_on_fire_count", "context",
]

# Label mapping: LIAR → FORGE internal labels
LABEL_MAP = {
    "pants-fire":   "fabricated",
    "false":        "misinfo",
    "barely-true":  "out_of_context",
    "half-true":    "out_of_context",
    "mostly-true":  "real",
    "true":         "real",
}

# ─── Download URLs (TSV files from HuggingFace dataset repo) ─────────────────
HF_BASE = "https://huggingface.co/datasets/liar/resolve/main/data"
SPLIT_URLS = {
    "train": f"{HF_BASE}/train.jsonl",
    "test":  f"{HF_BASE}/test.jsonl",
    "valid": f"{HF_BASE}/validation.jsonl",
}

# Fallback: direct GitHub mirror of the original TSV files
GITHUB_BASE = "https://raw.githubusercontent.com/v1shwa/fake-news-detection/master/LIAR_dataset"
TSV_URLS = {
    "train": f"{GITHUB_BASE}/train.tsv",
    "test":  f"{GITHUB_BASE}/test.tsv",
    "valid": f"{GITHUB_BASE}/valid.tsv",
}


def download_tsv_split(url: str, timeout: int = 30) -> list[dict]:
    """Download one TSV split, parse rows, return list of claim dicts."""
    import urllib.request
    logger.info("Fetching: %s", url)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return []

    rows = []
    reader = csv.reader(io.StringIO(raw), delimiter="\t")
    for row in reader:
        if len(row) < 2:
            continue
        # Pad row to expected column count
        padded = row + [""] * (len(COLUMNS) - len(row))
        d = dict(zip(COLUMNS, padded[:len(COLUMNS)]))
        forge_label = LABEL_MAP.get(d["label"].strip(), "misinfo")
        rows.append({
            "id":          d["id"],
            "statement":   d["statement"].strip(),
            "speaker":     d["speaker"].strip(),
            "party":       d["party"].strip(),
            "subject":     d["subject"].strip(),
            "liar_label":  d["label"].strip(),
            "forge_label": forge_label,
            "context":     d["context"].strip(),
        })
    return rows


def save_csv(rows: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "statement", "speaker", "party", "subject",
                  "liar_label", "forge_label", "context"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d claims to %s", len(rows), output_path)


def main():
    parser = argparse.ArgumentParser(description="Download the LIAR dataset")
    parser.add_argument(
        "--output", default="data/liar_dataset.csv",
        help="Output CSV path (default: data/liar_dataset.csv)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test", "valid"],
        help="Which splits to download (default: train test valid)"
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for split in args.splits:
        url = TSV_URLS.get(split)
        if not url:
            logger.warning("Unknown split: %s", split)
            continue
        rows = download_tsv_split(url)
        logger.info("  %s split: %d claims downloaded", split, len(rows))
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No data downloaded. Check your internet connection.")
        sys.exit(1)

    save_csv(all_rows, args.output)
    logger.info("Done! LIAR dataset ready at: %s", args.output)
    logger.info("Label distribution:")
    from collections import Counter
    for lbl, count in Counter(r["forge_label"] for r in all_rows).most_common():
        logger.info("  %s: %d", lbl, count)


if __name__ == "__main__":
    main()
