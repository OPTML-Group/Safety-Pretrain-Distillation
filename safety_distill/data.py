"""Datasets for safety distillation.

Two streams feed each training step:

* Distillation prompts (D_x): sequences from the safety-pretraining corpora.
  Only the first ``rollout_prefix_len`` tokens of each row are used, as the
  prompt from which the student samples its on-policy responses.
  - ``PackedTokenDataset`` / ``MultiSourceDataset`` read pre-tokenized uint16
    ``.bin`` files (same-tokenizer students).
  - ``RawTextDataset`` reads ``{"text": ...}`` jsonl and tokenizes with the
    student's tokenizer (cross-tokenizer students).
* Utility data (D_post): instruction/response pairs trained with a
  prompt-masked cross-entropy loss (``ChatSFTDataset``).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Memory-mapped uint16 token file viewed as ``[N, seq_len]`` rows."""

    def __init__(self, bin_path: str, seq_len: int):
        data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.n_seqs = data.shape[0] // seq_len
        self.seq_len = seq_len
        self.data = data[: self.n_seqs * seq_len].reshape(self.n_seqs, seq_len)

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx].astype(np.int64))


class MultiSourceDataset(Dataset):
    """Concatenation of several ``PackedTokenDataset`` files."""

    def __init__(self, paths: list[str], seq_len: int):
        self.datasets = [PackedTokenDataset(p, seq_len) for p in paths]
        self._lengths = [len(d) for d in self.datasets]
        self._offsets = np.cumsum([0] + self._lengths[:-1]).tolist()
        self._total = sum(self._lengths)
        self.source_counts = [
            (os.path.basename(p), n) for p, n in zip(paths, self._lengths)
        ]

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> torch.Tensor:
        for i, start in enumerate(self._offsets):
            if idx < start + self._lengths[i]:
                return self.datasets[i][idx - start]
        raise IndexError(f"index {idx} out of range for {self._total} sequences")


class RawTextDataset(Dataset):
    """Raw-text jsonl documents tokenized on the fly with the student tokenizer.

    Each item is a ``[seq_len]`` int64 tensor, truncated or right-padded with
    the pad token.
    """

    def __init__(self, jsonl_paths: list[str], tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        self.samples: list[str] = []
        self.source_counts: list[tuple[str, int]] = []
        for p in jsonl_paths:
            before = len(self.samples)
            with open(p) as f:
                for line in f:
                    if not line.strip():
                        continue
                    text = json.loads(line).get("text") or ""
                    if text:
                        self.samples.append(text)
            self.source_counts.append((os.path.basename(p), len(self.samples) - before))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.seq_len,
            add_special_tokens=False,
        )["input_ids"]
        if len(ids) < self.seq_len:
            ids = ids + [self.pad_token_id] * (self.seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class ChatSFTDataset(Dataset):
    """Instruction/response pairs formatted with the tokenizer's chat template.

    Expects a JSON list of ``{"instruction": str, "output": str}``. Prompt
    tokens and padding are masked with -100 so the loss covers only the
    response. The chat template already contains BOS and turn tokens, so
    tokenization uses ``add_special_tokens=False``.
    """

    ignore_index = -100

    def __init__(self, json_path: str, tokenizer, seq_len: int):
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("ChatSFTDataset requires a tokenizer with a chat_template")
        with open(json_path) as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        user = [{"role": "user", "content": sample["instruction"]}]
        prompt = self.tokenizer.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            user + [{"role": "assistant", "content": sample["output"]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_len = len(
            self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.seq_len,
                add_special_tokens=False,
            )["input_ids"]
        )
        labels[:prompt_len] = self.ignore_index
        labels[attention_mask == 0] = self.ignore_index
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def resolve_bin_sources(data_root: str, sources: list[str] | None) -> tuple[list[str], str | None]:
    """Train ``.bin`` paths and the tokenizer recorded in ``manifest.json``."""
    with open(os.path.join(data_root, "manifest.json")) as f:
        manifest = json.load(f)
    paths = [
        info["train_path"]
        for name, info in manifest["sources"].items()
        if sources is None or name in sources
    ]
    return paths, manifest.get("tokenizer")


def resolve_raw_text_sources(data_root: str, sources: list[str] | None) -> list[str]:
    """Train jsonl paths listed in ``raw_text_manifest.json``."""
    with open(os.path.join(data_root, "raw_text_manifest.json")) as f:
        manifest = json.load(f)
    return [
        info["train_path"]
        for name, info in manifest["sources"].items()
        if sources is None or name in sources
    ]
