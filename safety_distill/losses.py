"""Loss functions for safety distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def opd_topk_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    k: int = 16,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Top-k on-policy distillation loss (reverse KL on the student's top-k tokens).

    At every generated position t, let S_t be the student's k most probable
    tokens. The per-position term is

        sum_{v in S_t} w(v) * (log p_S(v) - log p_T(v)),
        w = softmax(log p_S restricted to S_t)  (detached),

    i.e. the reverse KL between the two distributions renormalized over S_t.

    The loss is written in policy-gradient form, as in PPO with one epoch:

        advantage = -(log p_S - log p_T) * w           (detached)
        ratio     = exp(log p_S - stop_grad(log p_S))  (= 1 numerically)
        loss      = -(advantage * ratio).sum(-1)

    which has the same value as ``(w * (log p_S - log p_T)).sum(-1)`` but the
    gradient of the on-policy objective. Differentiating the plain form
    instead would drop the KL factor from the gradient.

    Args:
        s_logits: student logits, ``(B, T, V)``; carries gradient.
        t_logits: teacher logits, ``(B, T, V)``; detached and cast to fp32.
        k: number of student candidate tokens per position.
        valid_mask: ``(B, T)`` mask of real generated positions (excludes
            padding after EOS). ``None`` averages over all positions.

    Returns:
        Scalar loss and a dict of intermediate tensors.
    """
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits.detach().float(), dim=-1)

    s_topk_vals, s_topk_ids = torch.topk(s_log_probs, k, dim=-1)
    t_at_s_topk = torch.gather(t_log_probs, dim=-1, index=s_topk_ids)
    kl_val = s_topk_vals - t_at_s_topk
    weights = F.softmax(s_topk_vals.detach(), dim=-1)

    advantage = (-kl_val * weights).detach()
    ratio = torch.exp(s_topk_vals - s_topk_vals.detach())
    per_token_loss = -(advantage * ratio).sum(dim=-1)
    if valid_mask is not None:
        mask = valid_mask.to(per_token_loss.dtype).detach()
        loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        loss = per_token_loss.mean()

    inner = {
        "s_topk_ids": s_topk_ids,
        "kl_val": kl_val,
        "weights": weights,
        "per_token_loss": per_token_loss,
    }
    return loss, inner


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy averaged over tokens whose label is not -100.

    Returns 0 (no gradient signal) when a batch has no supervised token, e.g.
    when the response is fully truncated.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    n_valid = (shift_labels != -100).sum()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    ) / n_valid.clamp(min=1)
