# src/data/build_processed_dataset.py

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import torch

from src.data.contract_builder import build_contract_sample


def find_pkl_files(root_dir: str) -> List[str]:
    root = Path(root_dir)
    return [str(p) for p in root.rglob("*.pkl")]


def load_raw_sample(pkl_path: str) -> Dict:
    with open(pkl_path, "rb") as f:
        # CogAge pickles are often saved with Python2/older numpy
        return pickle.load(f, encoding="latin1")


def main():
    # ------------------------------------------------------------
    # CHANGE THIS PATH: point it to your left-hand-data directory
    # Example:
    #   C:/Users/.../CogAge/python/dictionaries/left-hand-data/
    # ------------------------------------------------------------
    RAW_ROOT = "C:/Users/shefq/Documents/Datasets/CogAge/python/dictionaries/left-hand-data"

    # Output folder inside your project
    OUT_DIR = Path("processed/behavior_blho")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pkl_files = find_pkl_files(RAW_ROOT)
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No .pkl files found under: {RAW_ROOT}")

    samples = []
    skipped_non_blho = 0
    failed = 0

    for i, pkl_path in enumerate(pkl_files, start=1):
        try:
            raw = load_raw_sample(pkl_path)

            # BLHO filter: keep ONLY left-hand executions (rightHand=False)
            # If the key doesn't exist for some reason, we keep it (safer),
            # but in CogAge it should exist.
            if "rightHand" in raw and bool(raw["rightHand"]) is True:
                skipped_non_blho += 1
                continue

            sample = build_contract_sample(raw)
            samples.append(sample)

        except Exception as e:
            failed += 1
            print(f"[WARN] Failed on {pkl_path}\n  -> {type(e).__name__}: {e}")

        if i % 200 == 0:
            print(f"Processed {i}/{len(pkl_files)} files | kept={len(samples)} | skipped={skipped_non_blho} | failed={failed}")

    # Save samples
    out_samples = OUT_DIR / "samples.pt"
    torch.save(samples, out_samples)

    # Save meta stats
    meta = {
        "raw_root": RAW_ROOT,
        "num_files_found": len(pkl_files),
        "num_samples_kept": len(samples),
        "skipped_right_hand": skipped_non_blho,
        "failed_files": failed,
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDONE ✅")
    print(f"Saved: {out_samples}")
    print("Meta:", meta)


if __name__ == "__main__":
    main()
