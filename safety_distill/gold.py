"""Cross-tokenizer GOLD distillation loss for on-policy distillation.

Provenance: the core of this module is adapted from the TRL library,
``trl.experimental.gold`` at commit 576a5f6 (Apache-2.0), specifically
``trl/experimental/utils.py`` (byte-offset helpers) and
``trl/experimental/gold/gold_trainer.py`` (``build_teacher_inputs_from_texts``,
``generalized_jsd_loss``, ``ULDLoss``). Functions marked "from TRL" are copied
with only cosmetic changes; deviations are marked "[adapted]".

Design summary:
  * Sequence alignment uses tokenizer-native byte offsets: student and teacher
    tokens are grouped at the byte boundaries where both tokenizers agree.
  * Row convention follows the upstream trainer: UNSHIFTED logits. The row
    paired with completion token c_k is the logits at c_k's own position.
  * Teacher inputs are built by ``build_teacher_inputs_from_texts``: the prompt
    is re-tokenized with special tokens (trailing EOS stripped), the completion
    is encoded with byte offsets, and a teacher EOS is appended at
    ``(content_len, content_len)`` and later dropped by ``skip_teacher_eos``.
  * The loss is the GOLD hybrid: generalized JSD on the matched vocabulary
    (tokens whose string exists in both vocabularies) plus sorted-probability
    L1 (ULD) on the unmatched remainder, with adaptive weights equal to the
    matched fraction of the teacher vocabulary.

Adaptations relative to upstream (wrapper-level; the loss core is unchanged):
  * ``crossentropy_weight=0.0``: the cross-entropy anchor is applied by the
    caller's training objective, not inside this loss.
  * ``skip_student_eos=False``: the student completion stream passed in holds
    only valid (non-pad, non-EOS) rollout tokens, so there is no student EOS
    row to drop. ``skip_teacher_eos=True`` is kept.
  * Teacher logits are expected in float32 (cast by the caller).
  * Samples whose teacher byte-offset normalization fails are skipped and
    counted (``gold_trl_skipped``) instead of raising for the whole batch.
  * Per-sample diagnostic accumulators are recorded; they do not affect the
    loss value.

Gradients flow through the student logits only; the teacher side is detached.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Byte-offset helpers (from TRL: trl/experimental/utils.py)
# ---------------------------------------------------------------------------


def pad_byte_offsets(offsets: list[tuple[int, int]], target_length: int, padding_side: str) -> torch.Tensor:
    """Build a ``[target_length, 2]`` long tensor from ``(start, end)`` byte-offset tuples,
    padding with ``(0, 0)`` on the requested side."""
    offs = torch.tensor(offsets, dtype=torch.long).reshape(-1, 2)
    pad_len = target_length - offs.size(0)
    if pad_len <= 0:
        return offs
    pad_block = torch.zeros(pad_len, 2, dtype=torch.long)
    return torch.cat([pad_block, offs], dim=0) if padding_side == "left" else torch.cat([offs, pad_block], dim=0)


def is_byte_level_tokenizer(backend) -> bool:
    """Whether ``backend`` is a ByteLevel BPE tokenizer (Llama-3 family, SmolLM, Qwen, ...): its pieces are in
    byte-to-unicode space, one char per source byte. Detected via the pre-tokenizer / decoder repr."""
    return "ByteLevel" in repr(backend.pre_tokenizer) or "ByteLevel" in repr(backend.decoder)


def piece_byte_len(piece: str) -> int:
    """UTF-8 byte length of a ByteLevel BPE token piece: each char maps 1:1 to one source byte.

    Cross-tokenizer ULD targets ByteLevel BPE pairs (Llama-3, Qwen, SmolLM, Phi, Mistral v0.3+, ...); SentencePiece
    tokenizers are out of scope."""
    return len(piece)


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs], strict=True))


_BYTE_LEVEL_DECODER = {ch: b for b, ch in _bytes_to_unicode().items()}


def _byte_level_piece_len(piece: str, text_bytes: bytes, start: int) -> int | None:
    piece_bytes = []
    for ch in piece:
        if ch not in _BYTE_LEVEL_DECODER:
            return None
        piece_bytes.append(_BYTE_LEVEL_DECODER[ch])
    if piece_bytes and piece_bytes[0] == ord(" ") and text_bytes[start : start + 1] != b" ":
        piece_bytes = piece_bytes[1:]
    return len(piece_bytes)


def _split_repeated_byte_offsets(byte_offsets: list[tuple[int, int]], tokens: list[str]) -> list[tuple[int, int]]:
    """Split repeated char-derived spans for byte-fallback or byte-level tokens."""
    normalized = list(byte_offsets)
    i = 0
    while i < len(byte_offsets):
        j = i + 1
        while j < len(byte_offsets) and byte_offsets[j] == byte_offsets[i]:
            j += 1

        if j - i > 1:
            start, end = byte_offsets[i]
            piece_lengths = [piece_byte_len(token) for token in tokens[i:j]]
            if sum(piece_lengths) == end - start:
                cursor = start
                for offset_idx, length in enumerate(piece_lengths, start=i):
                    normalized[offset_idx] = (cursor, cursor + length)
                    cursor += length

        i = j
    return normalized


def _normalize_byte_offsets(
    byte_offsets: list[tuple[int, int]], tokens: list[str], text_bytes: bytes
) -> list[tuple[int, int]]:
    byte_offsets = _split_repeated_byte_offsets(byte_offsets, tokens)
    normalized = []
    cursor = 0

    for idx, (start, end) in enumerate(byte_offsets):
        if start == end:
            normalized.append((cursor, cursor))
            continue

        piece_len = _byte_level_piece_len(tokens[idx], text_bytes, start)
        next_start = byte_offsets[idx + 1][0] if idx + 1 < len(byte_offsets) else None
        has_overlap = start < cursor or (next_start is not None and next_start < end)

        if piece_len is not None and (has_overlap or piece_len == end - start):
            candidate_start = max(start, cursor)
            candidate_end = candidate_start + piece_len
            if candidate_end <= end:
                start = candidate_start
                end = candidate_end

        if start < cursor or end < start:
            raise ValueError(
                "Tokenizer produced overlapping byte offsets that could not be normalized. "
                "Cross-tokenizer ULD requires monotonic byte offsets."
            )

        normalized.append((start, end))
        cursor = end

    return normalized


def encode_with_byte_offsets(backend, texts: list[str], add_special_tokens: bool = False):
    """Encode ``texts`` and return per-text ``(ids, byte_offsets)`` pairs.

    Byte offsets are derived from the fast tokenizer's character offsets via an O(N) char-to-byte cumulative table.
    Overlapping spans from byte-level and byte-fallback tokens are split across their byte pieces."""
    if not is_byte_level_tokenizer(backend):
        raise NotImplementedError(
            "Cross-tokenizer ULD currently supports only ByteLevel BPE tokenizers "
            "(Llama-3, Qwen, SmolLM, Phi, Mistral v0.3+, ...). The given tokenizer is not ByteLevel."
        )
    encs = backend.encode_batch(texts, add_special_tokens=add_special_tokens)
    out = []
    for text, enc in zip(texts, encs, strict=True):
        char_to_byte = [0]
        for ch in text:
            char_to_byte.append(char_to_byte[-1] + len(ch.encode("utf-8")))
        byte_offsets = [(char_to_byte[s], char_to_byte[e]) for s, e in enc.offsets]
        byte_offsets = _normalize_byte_offsets(byte_offsets, enc.tokens, text.encode("utf-8"))
        out.append((list(enc.ids), byte_offsets))
    return out


# ---------------------------------------------------------------------------
# [adapted] Minimal replacement for trl.trainer.utils.pad (1-D sequences only):
# pad a list of 1-D tensors to a common length and stack them.
# ---------------------------------------------------------------------------


def _pad_1d(tensors: list[torch.Tensor], padding_side: str, padding_value: int) -> torch.Tensor:
    max_len = max(t.numel() for t in tensors)
    out = torch.full((len(tensors), max_len), padding_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        if padding_side == "left":
            out[i, max_len - t.numel():] = t
        else:
            out[i, : t.numel()] = t
    return out


# ---------------------------------------------------------------------------
# Teacher input construction (from TRL: gold_trainer.py; `pad` swapped for _pad_1d)
# ---------------------------------------------------------------------------


def build_teacher_inputs_from_texts(
    tokenizer,
    prompt_texts: list[str],
    completion_texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize teacher prompts/completions and produce tensors ready for GOLD loss.

    Returns ``(input_ids, labels, attention_mask, byte_offsets)``. ``byte_offsets`` is a ``[batch, seq, 2]`` tensor of
    UTF-8 byte ``(start, end)`` for each token: prompt and padding positions are filled with ``(0, 0)``; completion
    tokens carry offsets relative to the corresponding ``completion_text``; the appended EOS gets ``(content_len,
    content_len)``. Byte offsets are derived from the fast tokenizer's char offsets via ``encode_with_byte_offsets``.
    """

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    backend = tokenizer.backend_tokenizer

    prompt_token_ids = tokenizer(prompt_texts, add_special_tokens=True)["input_ids"]
    completion_encs = encode_with_byte_offsets(backend, completion_texts, add_special_tokens=False)

    sequences: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    offsets_list: list[list[tuple[int, int]]] = []

    for prompt_ids, (enc_ids, enc_offs), completion_text in zip(
        prompt_token_ids, completion_encs, completion_texts, strict=True
    ):
        # Remove trailing EOS from prompt so completions can extend cleanly
        if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
            prompt_ids = prompt_ids[:-1]

        completion_ids = list(enc_ids)
        completion_offs = list(enc_offs)
        content_len = len(completion_text.encode("utf-8"))

        sequence = list(prompt_ids) + completion_ids
        offsets = [(0, 0)] * len(prompt_ids) + completion_offs
        if eos_token_id is not None:
            sequence.append(eos_token_id)
            offsets.append((content_len, content_len))

        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        sequences.append(seq_tensor)
        attention_masks.append(torch.ones_like(seq_tensor))
        offsets_list.append(offsets)

        labels = seq_tensor.clone()
        labels[: len(prompt_ids)] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        labels_list.append(labels)

    teacher_input_ids = _pad_1d(
        sequences,
        padding_side="right",
        padding_value=pad_token_id if pad_token_id is not None else 0,
    )
    teacher_attention_mask = _pad_1d(attention_masks, padding_side="right", padding_value=0).bool()
    teacher_labels = _pad_1d(labels_list, padding_side="right", padding_value=-100)

    if eos_token_id is not None:
        for row in range(teacher_attention_mask.size(0)):
            valid = (
                teacher_input_ids[row] != pad_token_id
                if pad_token_id is not None
                else teacher_attention_mask[row].bool()
            )
            if valid.any():
                last_idx = valid.nonzero(as_tuple=True)[0][-1]
                teacher_attention_mask[row, last_idx + 1 :] = False

    target_len = teacher_input_ids.size(1)
    teacher_byte_offsets = torch.stack(
        [pad_byte_offsets(offs, target_len, padding_side="right") for offs in offsets_list],
        dim=0,
    )

    return teacher_input_ids, teacher_labels, teacher_attention_mask, teacher_byte_offsets


# ---------------------------------------------------------------------------
# Generalized JSD (from TRL: GOLDTrainer.generalized_jsd_loss, as a module function)
# ---------------------------------------------------------------------------


def generalized_jsd_loss(
    student_logits,
    teacher_logits,
    labels=None,
    beta=0.5,
    temperature=1.0,
    reduction="batchmean",
    logits_are_probs=False,
):
    """Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
    of https://huggingface.co/papers/2306.13649 for the definition."""

    if logits_are_probs:
        student_log_probs = torch.log(student_logits.clamp_min(1e-8))
        teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
    else:
        # Apply temperature scaling to logits before computing probabilities
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        # Compute the log of the mixture distribution
        # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
        beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
            dim=0,
        )

        # Compute KL divergences using F.kl_div
        # PyTorch's argument order is swapped relative to the paper's definition.
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

        # Compute the Generalized Jensen-Shannon Divergence
        jsd = beta * kl_teacher + (1 - beta) * kl_student

    # Masking
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    # Apply reduction
    if reduction == "batchmean":
        # [adapted] clamp_min(1) avoids 0/0 -> nan when no position is unmasked.
        denom = mask.sum().clamp_min(1) if labels is not None else max(jsd.size(0), 1)
        return jsd.sum() / denom
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    else:
        return jsd


# ---------------------------------------------------------------------------
# ULD / GOLD core (from TRL: class ULDLoss in gold_trainer.py). The config
# object is replaced by explicit keyword arguments; defaults follow this
# module's on-policy usage, and upstream defaults are noted where they differ.
# ---------------------------------------------------------------------------


class GoldTRLULD(nn.Module):
    """Universal Logit Distillation loss with the GOLD hybrid extension (TRL ULDLoss).

    Upstream defaults: skip_student_eos=True, skip_teacher_eos=True,
    use_hybrid_loss=False, beta=0.5, temperatures 1.0. This module defaults to
    skip_student_eos=False (see the module docstring) and use_hybrid_loss=True,
    which selects the GOLD method: generalized JSD on tokens with a 1:1
    vocabulary mapping and the ULD sorted-probability L1 on the rest. With
    use_hybrid_loss=False the loss reduces to plain ULD.
    """

    def __init__(
        self,
        student_tokenizer=None,
        teacher_tokenizer=None,
        *,
        crossentropy_weight: float = 0.0,
        distillation_weight: float = 1.0,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        skip_student_eos: bool = False,
        skip_teacher_eos: bool = True,
        use_extended_uld: bool = True,
        use_hybrid_loss: bool = True,
        hybrid_matched_weight: float | None = None,
        hybrid_unmatched_weight: float | None = None,
        beta: float = 0.5,
        device=None,
    ):
        super().__init__()
        self.device = device
        self.crossentropy_weight = crossentropy_weight
        self.distillation_weight = distillation_weight
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.use_extended_uld = use_extended_uld
        self.ignore_index = -100

        # Tokenizers for vocabulary alignment
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        # Hybrid ULD configuration
        self.use_hybrid_loss = use_hybrid_loss
        self.hybrid_matched_weight = hybrid_matched_weight
        self.hybrid_unmatched_weight = hybrid_unmatched_weight
        self.beta = beta  # JSD beta for matched tokens

        # Initialize vocabulary mapping for hybrid loss
        self._vocab_mapping = None
        self._teacher_matched_ids = None
        self._student_matched_ids = None
        if self.use_hybrid_loss and student_tokenizer is not None and teacher_tokenizer is not None:
            self._initialize_vocabulary_mapping()

        # [adapted] Per-sample diagnostic accumulators (logging only).
        self.last_n_groups_list: list[int] = []
        self.last_matched_list: list[float] = []
        self.last_unmatched_list: list[float] = []
        self.last_matched_loss = None
        self.last_unmatched_loss = None

    def __call__(
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
        student_byte_offsets=None,
        teacher_byte_offsets=None,
    ):
        """Compute the ULD loss with the GKD trainer interface."""
        # Cross-entropy loss for the student (disabled by default)
        if self.crossentropy_weight > 0:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = student_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            crossentropy_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            crossentropy_loss = self.crossentropy_weight * crossentropy_loss
        else:
            crossentropy_loss = 0.0

        # Distillation loss using the ULD approximation
        distillation_loss = self._compute_distillation_loss(
            student_logits,
            teacher_logits,
            student_labels,
            teacher_labels,
            student_input_ids,
            teacher_input_ids,
            student_byte_offsets=student_byte_offsets,
            teacher_byte_offsets=teacher_byte_offsets,
        )

        return crossentropy_loss + distillation_loss

    def _initialize_vocabulary_mapping(self):
        """Map teacher token ids to student token ids for identical token strings."""
        student_vocab = self.student_tokenizer.get_vocab()
        teacher_vocab = self.teacher_tokenizer.get_vocab()

        student_token_to_id = dict(student_vocab.items())

        vocab_mapping = {}
        teacher_matched_ids = set()
        student_matched_ids = set()

        for token_str, teacher_id in teacher_vocab.items():
            if token_str in student_token_to_id:
                student_id = student_token_to_id[token_str]
                vocab_mapping[teacher_id] = student_id
                teacher_matched_ids.add(teacher_id)
                student_matched_ids.add(student_id)

        self._vocab_mapping = vocab_mapping
        self._teacher_matched_ids = teacher_matched_ids
        self._student_matched_ids = student_matched_ids

        max_matched_teacher_id = max(self._vocab_mapping.keys())
        self.mapping_tensor = torch.full((max_matched_teacher_id + 1,), -1, dtype=torch.long)  # -1 for unmapped ids
        for k, v in self._vocab_mapping.items():
            self.mapping_tensor[k] = v
        if self.device is not None:
            self.mapping_tensor = self.mapping_tensor.to(self.device)

    def _compute_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
        student_byte_offsets=None,
        teacher_byte_offsets=None,
    ):
        """Per-sample aligned ULD / hybrid loss, averaged over the batch."""
        # [adapted] reset diagnostics for this batch
        self.last_n_groups_list = []
        self.last_matched_list = []
        self.last_unmatched_list = []

        # Answer regions
        student_answer_index, student_answer_size = self._get_start_and_size_answers(student_labels)
        teacher_answer_index, teacher_answer_size = self._get_start_and_size_answers(teacher_labels)

        if self.skip_student_eos:
            student_answer_size = [size - 1 for size in student_answer_size]
        if self.skip_teacher_eos:
            teacher_answer_size = [size - 1 for size in teacher_answer_size]

        # Edge case: all answer sizes are 0
        if (
            not student_answer_size
            or not teacher_answer_size
            or max(max(student_answer_size), max(teacher_answer_size)) <= 0
        ):
            return torch.zeros(1, device=student_logits.device, requires_grad=True) * student_logits.sum() * 1e-8

        batch_size = student_logits.size(0)
        distillation_losses = []

        for i in range(batch_size):
            student_start = student_answer_index[i]
            student_size = student_answer_size[i]
            teacher_start = teacher_answer_index[i]
            teacher_size = teacher_answer_size[i]

            if student_size <= 0 or teacher_size <= 0:
                loss_i = student_logits[i].sum() * 0.0
                distillation_losses.append(loss_i)
                continue

            # Answer logits
            student_answer_logits = student_logits[i, student_start : student_start + student_size]
            teacher_answer_logits = teacher_logits[i, teacher_start : teacher_start + teacher_size]

            # Probabilities
            student_probs = F.softmax(student_answer_logits / self.student_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_answer_logits / self.teacher_temperature, dim=-1)

            # Actual token ids, so split-token groups can multiply conditional probabilities.
            student_token_ids = student_input_ids[i, student_start : student_start + student_size].tolist()
            teacher_token_ids = teacher_input_ids[i, teacher_start : teacher_start + teacher_size].tolist()

            if self.use_extended_uld:
                if student_byte_offsets is None or teacher_byte_offsets is None:
                    raise ValueError("Byte offsets are required when `use_extended_uld=True`.")

                # Both sides are completion-relative, so plain slicing gives a shared byte coordinate system.
                s_answer = student_byte_offsets[i, student_start : student_start + student_size].tolist()
                t_answer = teacher_byte_offsets[i, teacher_start : teacher_start + teacher_size].tolist()
                student_groups, teacher_groups = self._align_by_byte_offsets(s_answer, t_answer)
                # Drop degenerate pairs where either side is empty (e.g. a trailing zero-width teacher EOS paired
                # with an empty student group; only reachable when ``skip_teacher_eos=False``).
                paired = [(sg, tg) for sg, tg in zip(student_groups, teacher_groups, strict=False) if sg and tg]
                student_groups = [sg for sg, _ in paired]
                teacher_groups = [tg for _, tg in paired]
                student_aligned = self._merge_probabilities_with_alignment_groups(
                    student_probs, student_groups, student_token_ids
                )
                teacher_aligned = self._merge_probabilities_with_alignment_groups(
                    teacher_probs, teacher_groups, teacher_token_ids
                )
                self.last_n_groups_list.append(len(student_groups))  # [adapted] diagnostics
            else:
                min_length = min(len(student_token_ids), len(teacher_token_ids))
                student_aligned = student_probs[:min_length, :]
                teacher_aligned = teacher_probs[:min_length, :]
                self.last_n_groups_list.append(min_length)  # [adapted] diagnostics

            if self.use_hybrid_loss and self._vocab_mapping is not None:
                # Hybrid: direct comparison for matched tokens, sorting for unmatched
                aligned_loss = self._compute_hybrid_uld_loss(student_aligned, teacher_aligned)
            else:
                # Plain ULD: sort all probabilities
                student_sorted = student_aligned.sort(dim=-1, descending=True).values
                teacher_sorted = teacher_aligned.sort(dim=-1, descending=True).values

                # Pad vocabularies to the same size
                student_vocab_size = student_sorted.size(-1)
                teacher_vocab_size = teacher_sorted.size(-1)
                max_vocab_size = max(student_vocab_size, teacher_vocab_size)

                if student_vocab_size < max_vocab_size:
                    student_sorted = F.pad(student_sorted, (0, max_vocab_size - student_vocab_size))
                if teacher_vocab_size < max_vocab_size:
                    teacher_sorted = F.pad(teacher_sorted, (0, max_vocab_size - teacher_vocab_size))

                # L1 distance (ULD)
                aligned_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
                aligned_loss /= student_aligned.size(0)  # Normalize by sequence length
            distillation_losses.append(aligned_loss)

        distillation_loss = torch.stack(distillation_losses).mean()
        return self.distillation_weight * distillation_loss

    @staticmethod
    def _align_by_byte_offsets(s_offsets, t_offsets):
        """Walk both byte-offset arrays, advancing the side whose current token ends earlier. A group closes when both
        sides reach the same byte boundary, i.e. where the two tokenizers agree on a split.
        """
        s_groups, t_groups = [], []
        s_start = t_start = s = t = 0
        n_s, n_t = len(s_offsets), len(t_offsets)
        while s < n_s and t < n_t:
            s_end, t_end = s_offsets[s][1], t_offsets[t][1]
            if s_end < t_end:
                s += 1
            elif s_end > t_end:
                t += 1
            else:
                s += 1
                t += 1
                s_groups.append(list(range(s_start, s)))
                t_groups.append(list(range(t_start, t)))
                s_start, t_start = s, t
        if s < n_s or t < n_t:
            s_groups.append(list(range(s_start, n_s)))
            t_groups.append(list(range(t_start, n_t)))
        return s_groups, t_groups

    def _merge_probabilities_with_alignment_groups(self, probs, alignment_groups, token_ids=None):
        """Merge each alignment group into one distribution.

        For a group merging tokens at positions [i, i+1, ..., i+k]:
            P_merged(y | x) = P(y | x) * P(token_{i+1} | token_i, x) * ... * P(token_{i+k} | ..., x)
        i.e. the marginal at the group's first position times the SCALAR conditionals of the actual generated tokens.
        Unnormalized, but preserves the joint probability of the actual sequence.
        """
        if not alignment_groups:
            return probs

        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device, dtype=probs.dtype)
        eps = 1e-8

        for group_idx, group in enumerate(alignment_groups):
            if len(group) > 1:
                if token_ids is None:
                    raise ValueError(
                        "token_ids must be provided when merging multi-token groups. "
                        "This is required for mathematically correct probability merging."
                    )

                # Marginal distribution at the first position
                first_pos = group[0]
                marginal_probs = probs[first_pos]

                # Scalar conditional probability of each subsequent actual token
                conditional_prob_product = 1.0
                for idx in group[1:]:
                    actual_token_id = token_ids[idx]
                    token_prob = probs[idx, actual_token_id].clamp_min(eps)
                    conditional_prob_product *= token_prob

                merged_probs = marginal_probs * conditional_prob_product
                aligned_probs[group_idx] = merged_probs

            elif len(group) == 1:
                aligned_probs[group_idx] = probs[group[0]]
            else:
                aligned_probs[group_idx] = torch.zeros_like(probs[0])

        return aligned_probs

    def _compute_hybrid_uld_loss(self, student_aligned, teacher_aligned):
        """GOLD hybrid loss.

        1. Generalized JSD on matched-vocabulary columns; 2. sort-pad-L1 on the complement;
        3. adaptive weights = matched_count / teacher_vocab_size unless fixed weights are given.
        """
        device = student_aligned.device
        student_vocab_size = student_aligned.size(-1)
        teacher_vocab_size = teacher_aligned.size(-1)

        # Sorted index tensors for the matched vocabulary
        if self._teacher_matched_ids:
            teacher_matched_indices = torch.tensor(sorted(self._teacher_matched_ids), dtype=torch.long, device=device)
            student_matched_indices = self.mapping_tensor[teacher_matched_indices]
        else:
            teacher_matched_indices = torch.tensor([], dtype=torch.long, device=device)
            student_matched_indices = torch.tensor([], dtype=torch.long, device=device)

        # Masks for matched tokens
        teacher_matched_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(student_vocab_size, dtype=torch.bool, device=device)

        if len(teacher_matched_indices) > 0:
            teacher_matched_mask[teacher_matched_indices] = True
            student_matched_mask[student_matched_indices] = True

        # 1. JSD loss for matched vocabulary tokens
        matched_loss = torch.tensor(0.0, device=device)
        matched_token_count = 0
        if len(teacher_matched_indices) > 0:
            teacher_matched_probs = teacher_aligned[:, teacher_matched_indices]  # [seq_len, num_matched]
            student_matched_probs = student_aligned[:, student_matched_indices]  # [seq_len, num_matched]
            matched_token_count = teacher_matched_probs.size(-1)

            matched_loss = self._compute_jsd_loss_for_matched_tokens(student_matched_probs, teacher_matched_probs)

        # 2. Sorted comparison loss for unmatched vocabulary tokens
        teacher_unmatched_mask = ~teacher_matched_mask
        student_unmatched_mask = ~student_matched_mask

        teacher_unmatched_probs = teacher_aligned[:, teacher_unmatched_mask]  # [seq_len, num_teacher_unmatched]
        student_unmatched_probs = student_aligned[:, student_unmatched_mask]  # [seq_len, num_student_unmatched]

        unmatched_loss = torch.tensor(0.0, device=device)
        if teacher_unmatched_probs.size(-1) > 0 and student_unmatched_probs.size(-1) > 0:
            teacher_unmatched_sorted = teacher_unmatched_probs.sort(dim=-1, descending=True).values
            student_unmatched_sorted = student_unmatched_probs.sort(dim=-1, descending=True).values

            # Pad to the same size if needed
            teacher_unmatched_size = teacher_unmatched_sorted.size(-1)
            student_unmatched_size = student_unmatched_sorted.size(-1)
            max_unmatched_size = max(teacher_unmatched_size, student_unmatched_size)

            if teacher_unmatched_size < max_unmatched_size:
                teacher_unmatched_sorted = F.pad(
                    teacher_unmatched_sorted, (0, max_unmatched_size - teacher_unmatched_size)
                )
            if student_unmatched_size < max_unmatched_size:
                student_unmatched_sorted = F.pad(
                    student_unmatched_sorted, (0, max_unmatched_size - student_unmatched_size)
                )

            # L1 loss on sorted unmatched tokens
            unmatched_loss = F.l1_loss(student_unmatched_sorted, teacher_unmatched_sorted, reduction="sum")
            unmatched_loss /= student_aligned.size(0)  # Normalize by sequence length

        # 3. Combine losses with weights
        if self.hybrid_matched_weight is None:
            # Adaptive weighting based on vocabulary overlap
            hybrid_matched_weight = matched_token_count / max(1, teacher_vocab_size)
            hybrid_unmatched_weight = 1.0 - hybrid_matched_weight
        else:
            hybrid_matched_weight = self.hybrid_matched_weight
            hybrid_unmatched_weight = self.hybrid_unmatched_weight

        total_loss = hybrid_matched_weight * matched_loss + hybrid_unmatched_weight * unmatched_loss

        # Store matched/unmatched components for logging
        self.last_matched_loss = matched_loss
        self.last_unmatched_loss = unmatched_loss
        self.last_matched_list.append(float(matched_loss.detach()))  # [adapted] diagnostics
        self.last_unmatched_list.append(float(unmatched_loss.detach()))  # [adapted] diagnostics

        return total_loss

    def _compute_jsd_loss_for_matched_tokens(self, student_logits, teacher_logits):
        """Generalized JSD on matched-token columns. Inputs are PROBABILITIES (logits_are_probs=True)."""
        _, num_matched = student_logits.shape

        student_logits_reshaped = student_logits.view(-1, num_matched)
        teacher_logits_reshaped = teacher_logits.view(-1, num_matched)

        jsd_loss = generalized_jsd_loss(
            student_logits_reshaped,
            teacher_logits_reshaped,
            labels=None,  # No masking needed for matched tokens
            beta=self.beta,
            temperature=1.0,  # Already applied upstream
            reduction="batchmean",
            logits_are_probs=True,
        )

        return jsd_loss

    def _get_start_and_size_answers(self, answer_tensors):
        """Start index and length of the non-ignored region of each label row."""
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            answer_mask = answer.ne(self.ignore_index)
            if not answer_mask.any():
                answers_index.append(0)
                answers_size.append(0)
                continue

            valid_indices = answer_mask.nonzero(as_tuple=True)[0]
            answers_index.append(int(valid_indices[0].item()))
            answers_size.append(int(answer_mask.sum().item()))
        return answers_index, answers_size


# ---------------------------------------------------------------------------
# On-policy wrapper
# ---------------------------------------------------------------------------


def student_completion_byte_offsets(student_tokenizer, gen_ids: list[int]) -> list[tuple[int, int]]:
    """Completion-relative byte offsets for the student's own sampled tokens: cumulative ``piece_byte_len`` over the
    completion tokens, with no decode/re-encode round trip (same logic as TRL
    ``GOLDTrainer._maybe_add_completion_byte_offsets``)."""
    offs: list[tuple[int, int]] = []
    cumulative = 0
    pieces = student_tokenizer.convert_ids_to_tokens(gen_ids)
    for piece in pieces:
        nb = piece_byte_len(piece)
        offs.append((cumulative, cumulative + nb))
        cumulative += nb
    return offs


def gold_trl_prepare_batch(
    student_tokenizer,
    teacher_tokenizer,
    full_ids_cpu: list[list[int]],
    K_pre: int,
    n_valid_list: list[int],
):
    """Build the GOLD/ULD loss inputs from a rollout batch.

    Per sample: decode prompt/completion as the upstream on-policy path does
    (skip_special_tokens=False, clean_up_tokenization_spaces=False), build
    teacher inputs via ``build_teacher_inputs_from_texts``, and compute student
    byte offsets via ``piece_byte_len``.

    Returns None if no sample is kept; otherwise a dict of CPU tensors over the
    kept samples plus ``kept`` (original indices) and ``n_skipped``.
    [adapted] Samples with an empty completion or whose teacher offsets fail
    normalization (upstream: batch-wide ValueError) are skipped and counted.
    """
    kept: list[int] = []
    n_skipped = 0
    prompt_texts: list[str] = []
    completion_texts: list[str] = []
    teacher_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    student_offsets: list[list[tuple[int, int]]] = []
    B = len(full_ids_cpu)

    for b in range(B):
        n_valid = n_valid_list[b]
        if n_valid <= 0:
            n_skipped += 1
            continue
        prompt_ids = full_ids_cpu[b][:K_pre]
        gen_ids = full_ids_cpu[b][K_pre : K_pre + n_valid]
        prompt_text = student_tokenizer.decode(
            prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        completion_text = student_tokenizer.decode(
            gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        try:
            t_ids, t_labels, t_attn, t_offs = build_teacher_inputs_from_texts(
                teacher_tokenizer, [prompt_text], [completion_text],
            )
        except ValueError:
            # Offset normalization failed for this rollout: skip and count.
            n_skipped += 1
            continue
        kept.append(b)
        prompt_texts.append(prompt_text)
        completion_texts.append(completion_text)
        teacher_tensors.append((t_ids[0], t_labels[0], t_attn[0], t_offs[0]))
        student_offsets.append(student_completion_byte_offsets(student_tokenizer, gen_ids))

    if not kept:
        return None

    t_pad_id = teacher_tokenizer.pad_token_id if teacher_tokenizer.pad_token_id is not None else 0
    teacher_input_ids = _pad_1d([t[0] for t in teacher_tensors], "right", t_pad_id)
    teacher_labels = _pad_1d([t[1] for t in teacher_tensors], "right", -100)
    teacher_attention_mask = _pad_1d(
        [t[2].long() for t in teacher_tensors], "right", 0
    ).bool()
    t_target_len = teacher_input_ids.size(1)
    teacher_byte_offsets = torch.stack(
        [pad_byte_offsets(t[3].tolist(), t_target_len, padding_side="right") for t in teacher_tensors], dim=0
    )

    return {
        "kept": kept,
        "n_skipped": n_skipped,
        "prompt_texts": prompt_texts,
        "completion_texts": completion_texts,
        "teacher_input_ids": teacher_input_ids,
        "teacher_labels": teacher_labels,
        "teacher_attention_mask": teacher_attention_mask,
        "teacher_byte_offsets": teacher_byte_offsets,
        "student_offsets": student_offsets,
    }


def gold_trl_xtok_loss(
    s_logits_uns,          # (B, T_gen, V_s) UNSHIFTED student logits: row p = logits at gen position p
    full_ids,              # (B, K_pre + T_gen) student token ids
    K_pre,
    valid_mask,            # (B, T_gen) gen tokens != pad
    prep,                  # output of gold_trl_prepare_batch (CPU tensors)
    teacher_logits,        # (B_kept, t_max, V_t) frozen teacher logits, float32, UNSHIFTED
    uld: GoldTRLULD,
):
    """GOLD loss on the student's own rollout.

    Assembles the ULD loss inputs (unshifted logits, unshifted labels,
    completion-relative byte offsets on both sides) and calls the core.
    Gradients flow through ``s_logits_uns`` only. Returns ``(loss, inner)``
    where ``inner`` holds scalar diagnostics.
    """
    device = s_logits_uns.device
    B, T_gen, _V_s = s_logits_uns.shape
    kept = prep["kept"]
    Bk = len(kept)

    # Student-side tensors over the kept samples (generation region only)
    s_logits_k = s_logits_uns[kept]                                    # (Bk, T_gen, V_s)
    s_input_ids = full_ids[kept][:, K_pre : K_pre + T_gen].to(device)  # (Bk, T_gen)
    s_labels = s_input_ids.clone()
    s_labels[~valid_mask[kept]] = -100                                 # answer region = valid gen tokens
    s_byte_offsets = torch.zeros(Bk, T_gen, 2, dtype=torch.long)
    for i, offs in enumerate(prep["student_offsets"]):
        if offs:
            s_byte_offsets[i, : len(offs)] = torch.tensor(offs, dtype=torch.long)
    s_byte_offsets = s_byte_offsets.to(device)

    loss = uld(
        student_logits=s_logits_k,
        teacher_logits=teacher_logits,
        student_labels=s_labels,
        teacher_labels=prep["teacher_labels"].to(device),
        student_input_ids=s_input_ids,
        teacher_input_ids=prep["teacher_input_ids"].to(device),
        student_byte_offsets=s_byte_offsets,
        teacher_byte_offsets=prep["teacher_byte_offsets"].to(device),
    )
    loss = loss.squeeze()  # the all-empty edge case returns shape (1,)

    ngl = uld.last_n_groups_list
    ml, ul = uld.last_matched_list, uld.last_unmatched_list
    inner = {
        "method": "gold_trl",
        "gold_n_groups": float(sum(ngl) / max(len(ngl), 1)) if ngl else 0.0,
        "gold_matched_loss": float(sum(ml) / max(len(ml), 1)) if ml else float("nan"),
        "gold_unmatched_loss": float(sum(ul) / max(len(ul), 1)) if ul else float("nan"),
        "gold_trl_skipped": float(prep["n_skipped"]),
    }
    return loss, inner
