"""
Prepare the distillation prompt corpus (D_x).

Streams three safety pretraining subsets from HuggingFace, tokenizes with the
given tokenizer, packs documents into fixed-length sequences, and writes the
token ids as uint16 .bin files plus a JSON manifest.

Datasets:
  - locuslab/safeweb        (config: score_5_rephrased)
  - locuslab/refuseweb      (config: score_5_refusal)
  - locuslab/moral_education (config: score_5_morals)

Target: ~100M tokens per dataset (~300M total).

Output layout (under --output_dir):
  safeweb_train.bin
  refuseweb_train.bin
  moral_education_train.bin
  safeweb_val.bin      (1M tokens held out per source)
  refuseweb_val.bin
  moral_education_val.bin
  manifest.json        (dtype, seq_len, token counts per file)

The packed sequences use an EOS-separated stream, then reshape to
[N, seq_len]. Tokens are stored as uint16, so the bin format requires a
vocabulary smaller than 65536 (e.g. SmolLM2, 49152 tokens). For a student
with a different or larger vocabulary, use --output_format raw_text, which
writes jsonl text that the trainer tokenizes with the student tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class SourceSpec:
    name: str
    repo: str
    config: str
    text_field: str
    target_tokens: int
    val_tokens: int


DEFAULT_SOURCES = [
    SourceSpec(
        name="safeweb",
        repo="locuslab/safeweb",
        config="score_5_rephrased",
        text_field="text",
        target_tokens=100_000_000,
        val_tokens=1_000_000,
    ),
    SourceSpec(
        name="refuseweb",
        repo="locuslab/refuseweb",
        config="score_5_refusal",
        text_field="text",
        target_tokens=100_000_000,
        val_tokens=1_000_000,
    ),
    SourceSpec(
        name="moral_education",
        repo="locuslab/moral_education",
        config="score_5_morals",
        text_field="text",
        target_tokens=100_000_000,
        val_tokens=1_000_000,
    ),
]


def stream_texts(spec: SourceSpec, cache_dir: str | None) -> Iterator[str]:
    """Yield text strings from a streaming HF dataset."""
    ds = load_dataset(
        spec.repo,
        spec.config,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    for row in ds:
        text = row.get(spec.text_field)
        if text is None:
            continue
        text = str(text).strip()
        if text:
            yield text


def tokenize_stream(
    text_iter: Iterable[str],
    tokenizer,
    eos_id: int,
) -> Iterator[int]:
    """Tokenize each document and emit token ids followed by EOS."""
    for text in text_iter:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        yield from ids
        yield eos_id


def collect_tokens(
    spec: SourceSpec,
    tokenizer,
    eos_id: int,
    seq_len: int,
    cache_dir: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect target_tokens for train and val_tokens for val.
    Packed into [N_train, seq_len] / [N_val, seq_len] arrays.
    Truncates to the last whole sequence.
    """
    total_needed = spec.target_tokens + spec.val_tokens
    buf = np.empty(total_needed + seq_len, dtype=np.uint16)  # small overshoot pad
    count = 0

    text_iter = stream_texts(spec, cache_dir=cache_dir)
    token_iter = tokenize_stream(text_iter, tokenizer, eos_id)

    pbar = tqdm(total=total_needed, desc=f"tokenize {spec.name}", unit="tok")
    for tok in token_iter:
        if tok >= 65536:
            raise ValueError(
                f"Token id {tok} exceeds the uint16 range; use --output_format raw_text."
            )
        buf[count] = tok
        count += 1
        if count % 10000 == 0:
            pbar.update(10000)
        if count >= total_needed:
            break
    pbar.update(count - pbar.n)
    pbar.close()

    if count < total_needed:
        print(
            f"  WARNING: {spec.name} only produced {count} tokens "
            f"(wanted {total_needed}). Using what we have."
        )

    # Truncate to multiple of seq_len for clean reshape.
    train_count = (spec.target_tokens // seq_len) * seq_len
    val_count = (spec.val_tokens // seq_len) * seq_len
    total_usable = min(count, train_count + val_count)

    # Guarantee val slice fits.
    if total_usable < val_count + seq_len:
        val_count = (total_usable // 2 // seq_len) * seq_len
        train_count = total_usable - val_count
    else:
        train_count = ((total_usable - val_count) // seq_len) * seq_len

    train_arr = buf[:train_count].reshape(-1, seq_len)
    val_arr = buf[train_count : train_count + val_count].reshape(-1, seq_len)

    return train_arr, val_arr


def save_bin(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # memmap-style flat dump; shape recorded in manifest.
    arr.astype(np.uint16).tofile(path)


def collect_raw_text(
    spec: SourceSpec,
    tokenizer,
    cache_dir: str | None,
) -> tuple[list[str], list[str], int, int]:
    """Collect target_tokens of raw-text docs for train and val_tokens for val.

    Uses the passed-in tokenizer as an accounting ruler to decide when we've
    gathered enough content — the actual text written to jsonl is raw (no
    tokenization), so the cross-tokenizer trainer can re-tokenize with its
    own student tokenizer. The resulting token count at training time will
    approximate `target_tokens`, with a bounded skew set by the tokenization
    ratio between this ruler and the student's tokenizer.

    Returns (train_docs, val_docs, train_tok_count, val_tok_count).
    """
    total_needed = spec.target_tokens + spec.val_tokens
    train_docs: list[str] = []
    val_docs: list[str] = []

    tok_count = 0
    pbar = tqdm(total=total_needed, desc=f"raw_text {spec.name}", unit="tok")
    for text in stream_texts(spec, cache_dir=cache_dir):
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        n = len(ids)
        # Allocate train first, then val.
        if tok_count < spec.target_tokens:
            train_docs.append(text)
            train_tok_count = min(n, spec.target_tokens - tok_count)
        else:
            val_docs.append(text)
            train_tok_count = 0
        tok_count += n
        if tok_count // 10000 > (tok_count - n) // 10000:
            pbar.update(n)
        if tok_count >= total_needed:
            break
    pbar.update(tok_count - pbar.n)
    pbar.close()

    # Track approximate token counts for manifest.
    train_tok_total = sum(
        len(tokenizer.encode(t, add_special_tokens=False)) for t in train_docs
    ) if train_docs else 0
    val_tok_total = sum(
        len(tokenizer.encode(t, add_special_tokens=False)) for t in val_docs
    ) if val_docs else 0
    return train_docs, val_docs, train_tok_total, val_tok_total


def save_jsonl(docs: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for text in docs:
            f.write(json.dumps({"text": text}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare safety distillation data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/distillation",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="HuggingFaceTB/SmolLM2-1.7B",
    )
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="*",
        default=None,
        help="Subset of source names to process (default: all three).",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="bin",
        choices=["bin", "raw_text"],
        help="'bin' (default) packs tokenized data into uint16 .bin files "
             "for students whose tokenizer matches --tokenizer. 'raw_text' "
             "writes one {\"text\": str} per line as .jsonl, tokenized at "
             "training time with the student tokenizer.",
    )
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, cache_dir=args.hf_cache_dir
    )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; cannot pack sequences.")

    sources = DEFAULT_SOURCES
    if args.sources:
        wanted = set(args.sources)
        sources = [s for s in sources if s.name in wanted]
        if not sources:
            raise ValueError(f"No sources matched filter {args.sources}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format == "bin":
        manifest: dict = {
            "tokenizer": args.tokenizer,
            "dtype": "uint16",
            "seq_len": args.seq_len,
            "eos_token_id": int(eos_id),
            "vocab_size": int(tokenizer.vocab_size),
            "sources": {},
        }

        for spec in sources:
            print(f"\n=== {spec.name} ({spec.repo} / {spec.config}) ===")
            train_arr, val_arr = collect_tokens(
                spec=spec,
                tokenizer=tokenizer,
                eos_id=eos_id,
                seq_len=args.seq_len,
                cache_dir=args.hf_cache_dir,
            )

            train_path = out_dir / f"{spec.name}_train.bin"
            val_path = out_dir / f"{spec.name}_val.bin"
            save_bin(train_arr, train_path)
            save_bin(val_arr, val_path)

            manifest["sources"][spec.name] = {
                "train_path": str(train_path),
                "val_path": str(val_path),
                "train_shape": list(train_arr.shape),
                "val_shape": list(val_arr.shape),
                "train_tokens": int(train_arr.size),
                "val_tokens": int(val_arr.size),
            }

            print(
                f"  wrote {train_path.name} shape={train_arr.shape} "
                f"({train_arr.size/1e6:.1f}M tokens)"
            )
            print(
                f"  wrote {val_path.name}   shape={val_arr.shape} "
                f"({val_arr.size/1e6:.1f}M tokens)"
            )

        manifest_path = out_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nmanifest: {manifest_path}")
    else:
        # raw_text: write {source}_train.jsonl + {source}_val.jsonl and a
        # separate raw_text_manifest.json, so both formats can share a directory.
        manifest = {
            "ruler_tokenizer": args.tokenizer,
            "format": "jsonl_raw_text",
            "sources": {},
        }
        for spec in sources:
            print(f"\n=== {spec.name} ({spec.repo} / {spec.config}) [raw_text] ===")
            train_docs, val_docs, train_tok, val_tok = collect_raw_text(
                spec=spec,
                tokenizer=tokenizer,
                cache_dir=args.hf_cache_dir,
            )
            train_path = out_dir / f"{spec.name}_train.jsonl"
            val_path = out_dir / f"{spec.name}_val.jsonl"
            save_jsonl(train_docs, train_path)
            save_jsonl(val_docs, val_path)
            manifest["sources"][spec.name] = {
                "train_path": str(train_path),
                "val_path": str(val_path),
                "train_docs": len(train_docs),
                "val_docs": len(val_docs),
                "train_tokens_ruler": int(train_tok),
                "val_tokens_ruler": int(val_tok),
            }
            print(
                f"  wrote {train_path.name}: {len(train_docs)} docs "
                f"({train_tok/1e6:.1f}M {args.tokenizer} tokens)"
            )
            print(
                f"  wrote {val_path.name}:   {len(val_docs)} docs "
                f"({val_tok/1e6:.1f}M {args.tokenizer} tokens)"
            )
        manifest_path = out_dir / "raw_text_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nraw_text_manifest: {manifest_path}")


if __name__ == "__main__":
    main()
