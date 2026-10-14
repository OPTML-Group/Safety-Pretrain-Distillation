"""
Prepare the safety post-training dataset (D_post) used by the utility loss.

Combines, following the post-training pipeline of Maini et al. 2025 (arXiv:2504.16980):
  - UltraChat-200k (instruction following)
  - WildGuardMix + WildJailbreak (safety instruction tuning)

Output: train.json and val.json as
  [{"instruction": "...", "output": "..."}, ...]
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset


def load_ultrachat(max_samples=None):
    """Load UltraChat-200k train_sft split and extract instruction-output pairs."""
    print("Loading UltraChat-200k...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    samples = []
    for row in ds:
        messages = row["messages"]
        # Extract consecutive user-assistant pairs
        for i in range(0, len(messages) - 1, 2):
            if messages[i]["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                instruction = messages[i]["content"].strip()
                output = messages[i + 1]["content"].strip()
                if instruction and output:
                    samples.append({"instruction": instruction, "output": output})

        if max_samples and len(samples) >= max_samples:
            samples = samples[:max_samples]
            break

    print(f"  UltraChat: {len(samples)} instruction-output pairs")
    return samples


def load_wildguardmix():
    """Load WildGuardMix with a 2x2 safety-label filter.

    WildGuardMix pairs matched refusal + compliance responses for each harmful
    prompt (built for classifier training, not SFT). Using it unfiltered teaches
    harmful compliance and over-refusal. Keep only the on-diagonal cells:
      - (prompt=harmful, response=refusal): teaches refusal
      - (prompt=unharmful, response=compliance, response_harm=unharmful): teaches helpful answering
    Drop everything else, including rows with missing labels.
    """
    print("Loading WildGuardMix...")
    ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")

    samples = []
    kept_harm_refuse = 0
    kept_benign_comply = 0
    dropped_harm_comply = 0
    dropped_benign_refuse = 0
    dropped_other = 0

    for row in ds:
        resp = row.get("response")
        if not (resp and resp.strip()):
            continue
        prompt = row["prompt"].strip()
        response = resp.strip()
        if not (prompt and response):
            continue

        p_harm = row.get("prompt_harm_label")
        r_refuse = row.get("response_refusal_label")
        r_harm = row.get("response_harm_label")

        if p_harm == "harmful" and r_refuse == "refusal":
            samples.append({"instruction": prompt, "output": response})
            kept_harm_refuse += 1
        elif (p_harm == "unharmful" and r_refuse == "compliance"
              and r_harm == "unharmful"):
            samples.append({"instruction": prompt, "output": response})
            kept_benign_comply += 1
        elif p_harm == "harmful" and r_refuse == "compliance":
            dropped_harm_comply += 1
        elif p_harm == "unharmful" and r_refuse == "refusal":
            dropped_benign_refuse += 1
        else:
            dropped_other += 1

    total_dropped = dropped_harm_comply + dropped_benign_refuse + dropped_other
    print(f"  WildGuardMix: kept {len(samples)} "
          f"(harmful+refusal={kept_harm_refuse}, "
          f"benign+compliance={kept_benign_comply})")
    print(f"    dropped {total_dropped} "
          f"(harmful+compliance={dropped_harm_comply}, "
          f"benign+refusal={dropped_benign_refuse}, "
          f"other/missing-labels={dropped_other})")
    return samples


def load_wildjailbreak(per_type_cap=None, seed=42):
    """Load WildJailbreak, bucket by data_type, optionally cap per bucket.

    Schema (HF card for allenai/wildjailbreak, config `train`):
      vanilla: str, adversarial: str, tactics: list, completion: str, data_type: str
      data_type values: vanilla_harmful | vanilla_benign
                        adversarial_harmful | adversarial_benign
      Counts: 50050 / 50050 / 82728 / 78706 (total 261,534).

    Route prompt field by data_type (vanilla_* -> vanilla, adversarial_* ->
    adversarial); completion is the shared target (refusal for harmful,
    compliance/continuation for benign by construction). Rows with unknown
    data_type are dropped (ablations in Jiang et al. 2024 Table 6 show all
    four types are needed).

    per_type_cap: if int, shuffle and truncate each bucket to this size.
      Default None = use all rows as-is. Set to 50000 to reproduce the WJB
      paper's §5.1 recipe ("sample 50K each of vanilla harmful, adversarial
       harmful, vanilla benign, and adversarial benign items").
    """
    print("Loading WildJailbreak...")
    try:
        ds = load_dataset("allenai/wildjailbreak", "train", split="train",
                          delimiter="\t", keep_default_na=False)
        rows = ds
    except Exception as e:
        print(f"  datasets.load_dataset failed ({e}), using pandas fallback...")
        import pandas as pd
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="allenai/wildjailbreak",
            filename="train/train.tsv",
            repo_type="dataset",
        )
        df = pd.read_csv(path, sep="\t", dtype=str)
        df = df.fillna("")
        rows = df.to_dict("records")

    buckets = {
        "vanilla_harmful": [],
        "vanilla_benign": [],
        "adversarial_harmful": [],
        "adversarial_benign": [],
    }
    unknown_type = 0
    empty_completion = 0
    empty_instruction = 0

    for row in rows:
        dt = str(row.get("data_type", "")).strip()
        if dt not in buckets:
            unknown_type += 1
            continue

        completion = str(row.get("completion", "")).strip()
        if not completion:
            empty_completion += 1
            continue

        if dt.startswith("vanilla"):
            instruction = str(row.get("vanilla", "")).strip()
        else:
            instruction = str(row.get("adversarial", "")).strip()
        if not instruction:
            empty_instruction += 1
            continue

        buckets[dt].append({"instruction": instruction, "output": completion})

    rng = random.Random(seed)
    samples = []
    for dt in ("vanilla_harmful", "vanilla_benign",
               "adversarial_harmful", "adversarial_benign"):
        items = buckets[dt]
        available = len(items)
        if per_type_cap is not None and available > per_type_cap:
            rng.shuffle(items)
            items = items[:per_type_cap]
        samples.extend(items)
        print(f"  WildJailbreak {dt}: kept {len(items)} / {available}")

    cap_note = f" (per_type_cap={per_type_cap})" if per_type_cap is not None else ""
    print(f"  WildJailbreak total: {len(samples)}{cap_note} "
          f"(unknown_type={unknown_type}, empty_completion={empty_completion}, "
          f"empty_instruction={empty_instruction})")
    return samples


def combine_datasets(ultrachat, wildguard, wildjailbreak, seed=42):
    """Combine and shuffle all datasets (WildGuardMix and WildJailbreak in full)."""
    rng = random.Random(seed)

    n_instruction = len(ultrachat)
    n_safety = len(wildguard) + len(wildjailbreak)

    print(f"\nCombining datasets:")
    print(f"  UltraChat (instruction): {n_instruction}")
    print(f"  WildGuardMix (safety):   {len(wildguard)}")
    print(f"  WildJailbreak (safety):  {len(wildjailbreak)}")
    print(f"  Total safety:            {n_safety}")
    print(f"  Total:                   {n_instruction + n_safety}")
    print(f"  Safety fraction:         {n_safety / (n_instruction + n_safety):.1%}")

    combined = ultrachat + wildguard + wildjailbreak
    rng.shuffle(combined)

    return combined


def split_and_save(data, output_dir, val_fraction=0.05, seed=42):
    """Split data into train/val and save as JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rng.shuffle(data)

    val_size = int(len(data) * val_fraction)
    val_data = data[:val_size]
    train_data = data[val_size:]

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=None)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=None)

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_data)} samples)")
    print(f"  Val:   {val_path} ({len(val_data)} samples)")

    return train_path, val_path


def estimate_tokens(data, avg_tokens_per_sample=400):
    """Rough token count estimate."""
    total = len(data) * avg_tokens_per_sample
    print(f"\nEstimated total tokens: ~{total / 1e6:.1f}M (assuming ~{avg_tokens_per_sample} tokens/sample)")
    return total


def main():
    parser = argparse.ArgumentParser(description="Prepare safety post-training dataset")
    parser.add_argument("--output_dir", type=str, default="data/safety_posttrain",
                        help="Output directory for train.json and val.json")
    parser.add_argument("--max_ultrachat", type=int, default=None,
                        help="Max UltraChat samples (None = use all)")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Validation split fraction (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Load datasets
    ultrachat = load_ultrachat(max_samples=args.max_ultrachat)
    wildguard = load_wildguardmix()
    wildjailbreak = load_wildjailbreak()

    combined = combine_datasets(ultrachat, wildguard, wildjailbreak, seed=args.seed)

    # Estimate token count
    estimate_tokens(combined)

    # Split and save
    split_and_save(combined, args.output_dir, val_fraction=args.val_fraction,
                   seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
