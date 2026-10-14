"""Chat templates for base-model students.

Distillation and the utility loss run in chat format, so the distilled
checkpoint is directly usable as a chat model. A base student without a
chat template receives its family's instruct template at training time:

* Llama-3 base models: the header/turn tokens are already single tokens in
  the base vocabulary, so the Llama-3 Instruct template is installed without
  resizing embeddings.
* SmolLM2 base models: the SmolLM2-Instruct ChatML template.
"""

from __future__ import annotations

import json
import os
import shutil

from huggingface_hub import snapshot_download

LLAMA3_INSTRUCT_REPO = "meta-llama/Llama-3.2-3B-Instruct"
SMOLLM2_INSTRUCT_REPO = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SMOLLM2_VOCAB_SIZE = 49152
SMOLLM2_EOS_TOKEN = "<|endoftext|>"
SMOLLM2_EOS_ID = 0


def _instruct_chat_template(repo_id: str) -> str:
    snap = snapshot_download(repo_id=repo_id, allow_patterns=["tokenizer_config.json"])
    with open(os.path.join(snap, "tokenizer_config.json")) as f:
        return json.load(f)["chat_template"]


def is_llama3_tokenizer(tokenizer) -> bool:
    """True if the Llama-3 header/turn special tokens exist as single tokens."""
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tid == unk:
            return False
        if len(tokenizer.encode(tok, add_special_tokens=False)) != 1:
            return False
    return True


def is_smollm2_tokenizer(tokenizer) -> bool:
    return getattr(tokenizer, "vocab_size", None) == SMOLLM2_VOCAB_SIZE


def ensure_chat_template(tokenizer) -> str:
    """Install the student family's chat template if the tokenizer has none.

    Returns a short description of what was done. Raises for an unknown
    family rather than guessing a format.
    """
    if getattr(tokenizer, "chat_template", None):
        return "existing chat template"
    if is_llama3_tokenizer(tokenizer):
        tokenizer.chat_template = _instruct_chat_template(LLAMA3_INSTRUCT_REPO)
        return "installed Llama-3 Instruct chat template"
    if is_smollm2_tokenizer(tokenizer):
        tokenizer.chat_template = _instruct_chat_template(SMOLLM2_INSTRUCT_REPO)
        return "installed SmolLM2-Instruct ChatML template"
    raise RuntimeError(
        "The student tokenizer has no chat template and is neither a Llama-3 nor a "
        "SmolLM2 tokenizer; set tokenizer.chat_template before training."
    )


def write_smollm2_package(out_dir: str) -> None:
    """Make a saved SmolLM2-1.7B student a complete chat-model package.

    Copies the SmolLM2-Instruct tokenizer files (chat template and special
    tokens), sets the EOS token to ``<|endoftext|>`` (id 0), and writes a
    ``generation_config.json`` that stops generation at that token. Model
    weights and ``config.json`` are not touched.
    """
    snap = snapshot_download(
        repo_id=SMOLLM2_INSTRUCT_REPO,
        allow_patterns=["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    )
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = os.path.join(snap, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))

    for fname in ("tokenizer_config.json", "special_tokens_map.json"):
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            cfg = json.load(f)
        cfg["eos_token"] = SMOLLM2_EOS_TOKEN
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    gen_cfg = {
        "_from_model_config": True,
        "bos_token_id": SMOLLM2_EOS_ID,
        "eos_token_id": SMOLLM2_EOS_ID,
    }
    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2, ensure_ascii=False)
