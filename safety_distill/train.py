"""Safety distillation from a safety-pretrained teacher.

Each optimizer step combines two terms:

    L = alpha * L_distill + (1 - alpha) * L_util

* ``L_distill`` is on-policy distillation. The student samples responses to
  prompts from the safety-pretraining corpora (D_x); the frozen teacher scores
  the student-generated tokens.
  - Same tokenizer: top-k reverse KL over the student's k most probable tokens
    (``losses.opd_topk_loss``).
  - Different tokenizers: GOLD, which aligns the two tokenizations by byte
    offsets and compares matched tokens with a JS divergence and unmatched
    tokens with the ULD distance (``gold.py``).
* ``L_util`` is the cross-entropy on safety post-training instruction data
  (D_post), formatted with the student's chat template. The distilled model
  is therefore a chat model; no separate instruction-tuning stage follows.

The student is trained in fp32; the teacher is held in bf16 and its logits
are cast to fp32 before any loss computation.

Example (single node, 4 GPUs, DeepSpeed ZeRO-2)::

    accelerate launch --use_deepspeed --zero_stage 2 --num_processes 4 \\
        --mixed_precision no --gradient_accumulation_steps 16 --gradient_clipping 1.0 \\
        -m safety_distill.train \\
        --teacher_model locuslab/safelm-1.7b \\
        --student_model HuggingFaceTB/SmolLM2-1.7B \\
        --distill_data_root data/distillation \\
        --posttrain_json data/safety_posttrain/train.json \\
        --alpha 0.7 --max_steps 3000 --output_dir checkpoints/smollm2_distill
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType, set_seed
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    get_cosine_schedule_with_warmup,
)

from .chat_format import ensure_chat_template, is_smollm2_tokenizer, write_smollm2_package
from .data import (
    ChatSFTDataset,
    MultiSourceDataset,
    RawTextDataset,
    resolve_bin_sources,
    resolve_raw_text_sources,
)
from .losses import masked_ce_loss, opd_topk_loss


@dataclass
class DistillConfig:
    # models
    teacher_model: str = "locuslab/safelm-1.7b"
    student_model: str = "HuggingFaceTB/SmolLM2-1.7B"
    cache_dir: str | None = None

    # data
    distill_data_root: str = ""          # holds manifest.json and/or raw_text_manifest.json
    sources: list[str] | None = None     # subset of D_x sources; None = all
    posttrain_json: str = ""             # [{"instruction", "output"}, ...]
    posttrain_max_seq_length: int = 512

    # rollouts
    rollout_prefix_len: int = 256
    rollout_gen_len: int = 256
    n_responses: int = 2
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0

    # loss
    alpha: float = 0.7
    top_k: int = 16                      # same-tokenizer OPD
    gold_beta: float = 0.5               # GOLD JS interpolation on matched tokens
    gold_matched_weight: float | None = None    # None = adaptive weights
    gold_unmatched_weight: float | None = None

    # optimization
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    global_batch_size: int = 64
    micro_batch_size: int = 1
    max_steps: int = 1500
    gradient_checkpointing: bool = True
    seed: int = 42

    # output
    output_dir: str = "checkpoints/safety_distill"
    save_every_steps: int = 500
    log_every_steps: int = 10


class _FiniteLogits(LogitsProcessor):
    """Replace non-finite logits during sampling; a no-op for finite logits."""

    def __call__(self, input_ids, scores):
        if torch.isfinite(scores).all():
            return scores
        return torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)


@contextlib.contextmanager
def _load_outside_zero3():
    """Load a frozen model whole while a DeepSpeed ZeRO-3 session is active.

    Under ZeRO-3, ``from_pretrained`` partitions parameters by default. The
    teacher never passes through ``accelerator.prepare``, so it must be loaded
    unpartitioned. No-op without ZeRO-3.
    """
    import transformers.integrations.deepspeed as hf_ds

    name = next(
        (n for n in ("_hf_deepspeed_config_weak_ref", "_hf_deepspeed_config_weakref")
         if hasattr(hf_ds, n)),
        None,
    )
    if name is None:
        yield
        return
    prev = getattr(hf_ds, name)
    setattr(hf_ds, name, None)
    try:
        yield
    finally:
        setattr(hf_ds, name, prev)


class SafetyDistillTrainer:
    def __init__(self, cfg: DistillConfig):
        self.cfg = cfg
        world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        if cfg.global_batch_size % (cfg.micro_batch_size * world_size):
            raise ValueError("global_batch_size must be divisible by micro_batch_size * world_size")
        self.accelerator = Accelerator(
            mixed_precision="no",
            gradient_accumulation_steps=cfg.global_batch_size // (cfg.micro_batch_size * world_size),
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=4))],
        )
        set_seed(cfg.seed)
        self.global_step = 0
        self.out_dir = Path(cfg.output_dir)

        self._load_models()
        self._load_data()
        self._setup_optimizer()

    # ----- setup -----

    def _load_models(self) -> None:
        cfg, acc = self.cfg, self.accelerator
        self.student_tokenizer = AutoTokenizer.from_pretrained(cfg.student_model, cache_dir=cfg.cache_dir)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model, cache_dir=cfg.cache_dir)
        for tok in (self.student_tokenizer, self.teacher_tokenizer):
            if tok.pad_token_id is None:
                tok.pad_token_id = tok.eos_token_id
        self.cross_tokenizer = (
            self.student_tokenizer.get_vocab() != self.teacher_tokenizer.get_vocab()
        )
        acc.print(f"[tokenizer] {'cross-tokenizer (GOLD)' if self.cross_tokenizer else 'shared (top-k OPD)'}")
        acc.print(f"[chat] {ensure_chat_template(self.student_tokenizer)}")

        with _load_outside_zero3():
            self.teacher = AutoModelForCausalLM.from_pretrained(
                cfg.teacher_model, torch_dtype=torch.bfloat16, cache_dir=cfg.cache_dir
            )
        self.teacher.eval().requires_grad_(False)

        self.student = AutoModelForCausalLM.from_pretrained(
            cfg.student_model, torch_dtype=torch.float32, cache_dir=cfg.cache_dir
        )
        if cfg.gradient_checkpointing:
            self.student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if self.cross_tokenizer:
            from .gold import GoldTRLULD

            self.gold = GoldTRLULD(
                student_tokenizer=self.student_tokenizer,
                teacher_tokenizer=self.teacher_tokenizer,
                beta=cfg.gold_beta,
                use_hybrid_loss=True,
                hybrid_matched_weight=cfg.gold_matched_weight,
                hybrid_unmatched_weight=cfg.gold_unmatched_weight,
                device=acc.device,
            )
            n_matched = len(self.gold._vocab_mapping or {})
            n_teacher = len(self.teacher_tokenizer.get_vocab())
            acc.print(f"[gold] {n_matched} matched tokens ({100 * n_matched / n_teacher:.2f}% of teacher vocab)")

    def _load_data(self) -> None:
        cfg, acc = self.cfg, self.accelerator
        seq_len = cfg.rollout_prefix_len + cfg.rollout_gen_len
        if not self.cross_tokenizer:
            # Pre-tokenized ids are only meaningful for the tokenizer that wrote them.
            paths, bin_tokenizer = resolve_bin_sources(cfg.distill_data_root, cfg.sources)
            if bin_tokenizer is not None:
                ref = AutoTokenizer.from_pretrained(bin_tokenizer, cache_dir=cfg.cache_dir)
                if ref.get_vocab() != self.student_tokenizer.get_vocab():
                    raise ValueError(
                        f"The .bin corpus was tokenized with {bin_tokenizer}, which does not "
                        f"match the student tokenizer; prepare raw-text data instead."
                    )
            self.prompt_dataset = MultiSourceDataset(paths, seq_len)
        else:
            paths = resolve_raw_text_sources(cfg.distill_data_root, cfg.sources)
            self.prompt_dataset = RawTextDataset(paths, self.student_tokenizer, seq_len)
        if not paths:
            raise ValueError(f"no distillation sources found under {cfg.distill_data_root}")
        for name, n in self.prompt_dataset.source_counts:
            acc.print(f"[data] D_x {name}: {n:,} rows")

        self.prompt_loader = DataLoader(
            self.prompt_dataset,
            batch_size=cfg.micro_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.posttrain_loader = None
        if cfg.alpha < 1.0:
            posttrain = ChatSFTDataset(
                cfg.posttrain_json, self.student_tokenizer, cfg.posttrain_max_seq_length
            )
            acc.print(f"[data] D_post: {len(posttrain):,} examples")
            self.posttrain_loader = DataLoader(
                posttrain,
                batch_size=cfg.micro_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
            )
        self.posttrain_iter = None

    def _setup_optimizer(self) -> None:
        cfg, acc = self.cfg, self.accelerator
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        # The prepared scheduler advances once per process at every optimizer
        # step, so its horizon is scaled by the number of processes.
        n_procs = acc.num_processes
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(cfg.max_steps * cfg.warmup_ratio) * n_procs,
            num_training_steps=cfg.max_steps * n_procs,
        )
        self.student, self.optimizer, self.prompt_loader, self.scheduler = acc.prepare(
            self.student, self.optimizer, self.prompt_loader, self.scheduler
        )
        if self.posttrain_loader is not None:
            self.posttrain_loader = acc.prepare(self.posttrain_loader)
        self.teacher = self.teacher.to(acc.device)

    # ----- helpers -----

    def _is_deepspeed(self) -> bool:
        return self.accelerator.distributed_type == DistributedType.DEEPSPEED

    def _is_zero3(self) -> bool:
        if not self._is_deepspeed():
            return False
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        return plugin is not None and int(plugin.zero_stage) == 3

    @torch.no_grad()
    def _rollout(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Sample ``n_responses`` student continuations per prefix.

        Returns ``(n_responses * B, prefix_len + gen_len)`` token ids, right-
        padded with the pad token after EOS.
        """
        cfg = self.cfg
        if self._is_zero3():
            import deepspeed

            gather = deepspeed.zero.GatheredParameters(
                list(self.accelerator.unwrap_model(self.student).parameters()), modifier_rank=None
            )
        else:
            gather = contextlib.nullcontext()
        with gather:
            model = self.accelerator.unwrap_model(self.student)
            # Gradient checkpointing disables the KV cache; turn it off for sampling.
            if cfg.gradient_checkpointing:
                model.gradient_checkpointing_disable()
            model.eval()
            try:
                out = model.generate(
                    input_ids=prefix_ids,
                    max_new_tokens=cfg.rollout_gen_len,
                    do_sample=True,
                    temperature=cfg.rollout_temperature,
                    top_p=cfg.rollout_top_p,
                    num_return_sequences=cfg.n_responses,
                    pad_token_id=self.student_tokenizer.pad_token_id,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([_FiniteLogits()]),
                )
            finally:
                model.train()
                if cfg.gradient_checkpointing:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
        total = cfg.rollout_prefix_len + cfg.rollout_gen_len
        if out.size(1) < total:
            pad = out.new_full((out.size(0), total - out.size(1)), self.student_tokenizer.pad_token_id)
            out = torch.cat([out, pad], dim=1)
        return out[:, :total]

    def _distill_loss(self, full_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        k_pre = cfg.rollout_prefix_len
        total = full_ids.size(1)
        gen_tokens = full_ids[:, k_pre:total]
        valid_mask = gen_tokens != self.student_tokenizer.pad_token_id
        student_logits = self.student(input_ids=full_ids).logits

        if not self.cross_tokenizer:
            with torch.no_grad():
                teacher_logits = self.teacher(input_ids=full_ids).logits
            # Row t-1 holds the distribution that generated token t.
            loss, _ = opd_topk_loss(
                student_logits[:, k_pre - 1 : total - 1, :],
                teacher_logits[:, k_pre - 1 : total - 1, :],
                k=min(cfg.top_k, student_logits.size(-1)),
                valid_mask=valid_mask,
            )
            return loss, {}

        from .gold import gold_trl_prepare_batch, gold_trl_xtok_loss

        # GOLD follows the TRL convention of unshifted rows.
        s_logits = student_logits[:, k_pre:total, :]
        prep = gold_trl_prepare_batch(
            self.student_tokenizer,
            self.teacher_tokenizer,
            full_ids.detach().cpu().tolist(),
            k_pre,
            [int(n) for n in valid_mask.sum(dim=1).tolist()],
        )
        if prep is None:
            return s_logits.sum() * 0.0, {"gold_skipped": float(full_ids.size(0))}
        with torch.no_grad():
            teacher_logits = self.teacher(
                input_ids=prep["teacher_input_ids"].to(full_ids.device),
                attention_mask=prep["teacher_attention_mask"].long().to(full_ids.device),
            ).logits.float()
        loss, inner = gold_trl_xtok_loss(
            s_logits_uns=s_logits,
            full_ids=full_ids,
            K_pre=k_pre,
            valid_mask=valid_mask,
            prep=prep,
            teacher_logits=teacher_logits,
            uld=self.gold,
        )
        return loss, {"gold_skipped": float(inner.get("gold_trl_skipped", 0.0) or 0.0)}

    def _next_posttrain_batch(self) -> dict:
        if self.posttrain_iter is None:
            self.posttrain_iter = iter(self.posttrain_loader)
        try:
            return next(self.posttrain_iter)
        except StopIteration:
            self.posttrain_iter = iter(self.posttrain_loader)
            return next(self.posttrain_iter)

    def _util_loss(self) -> torch.Tensor:
        batch = self._next_posttrain_batch()
        logits = self.student(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
        return masked_ce_loss(logits, batch["labels"])

    def _save(self, name: str) -> None:
        acc = self.accelerator
        ckpt_dir = self.out_dir / name
        if acc.is_main_process:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        acc.wait_for_everyone()
        state_dict = acc.get_state_dict(self.student) if self._is_zero3() else None
        if acc.is_main_process:
            model = acc.unwrap_model(self.student)
            model.save_pretrained(ckpt_dir, safe_serialization=True, state_dict=state_dict)
            self.student_tokenizer.save_pretrained(ckpt_dir)
            if is_smollm2_tokenizer(self.student_tokenizer):
                write_smollm2_package(str(ckpt_dir))
            with open(ckpt_dir / "training_config.json", "w") as f:
                json.dump({"global_step": self.global_step, "config": asdict(self.cfg)}, f, indent=2)
            acc.print(f"saved {ckpt_dir}")
        acc.wait_for_everyone()

    # ----- training -----

    def train(self) -> None:
        cfg, acc = self.cfg, self.accelerator
        alpha = cfg.alpha
        acc.print(
            f"alpha={alpha} lr={cfg.learning_rate} global_batch={cfg.global_batch_size} "
            f"grad_accum={acc.gradient_accumulation_steps} max_steps={cfg.max_steps}"
        )
        self.student.train()
        k_pre = cfg.rollout_prefix_len
        t0 = time.time()
        while self.global_step < cfg.max_steps:
            for prompt_ids in self.prompt_loader:
                with acc.accumulate(self.student):
                    full_ids = self._rollout(prompt_ids[:, :k_pre].contiguous())
                    distill, info = self._distill_loss(full_ids)
                    if alpha < 1.0:
                        util = self._util_loss()
                        loss = alpha * distill + (1.0 - alpha) * util
                    else:
                        util = torch.zeros((), device=distill.device)
                        loss = distill
                    acc.backward(loss)
                    # Under DeepSpeed the engine clips with --gradient_clipping.
                    if cfg.grad_clip > 0 and acc.sync_gradients and not self._is_deepspeed():
                        acc.clip_grad_norm_(self.student.parameters(), cfg.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if not acc.sync_gradients:
                    continue
                self.global_step += 1
                if self.global_step % cfg.log_every_steps == 0:
                    step_s = (time.time() - t0) / cfg.log_every_steps
                    extra = "".join(f" {k}={v:.3f}" for k, v in info.items())
                    acc.print(
                        f"[step {self.global_step}/{cfg.max_steps}] loss={loss.item():.4f} "
                        f"distill={distill.item():.4f} util_ce={util.item():.4f}{extra} "
                        f"lr={self.scheduler.get_last_lr()[0]:.2e} step_s={step_s:.1f}"
                    )
                    t0 = time.time()
                if self.global_step % cfg.save_every_steps == 0:
                    self._save(f"step_{self.global_step}")
                if self.global_step >= cfg.max_steps:
                    break
        if self.global_step % cfg.save_every_steps:
            self._save(f"step_{self.global_step}")
        acc.print("training complete")


def parse_args() -> DistillConfig:
    defaults = DistillConfig()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--teacher_model", default=defaults.teacher_model)
    p.add_argument("--student_model", default=defaults.student_model)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--distill_data_root", required=True,
                   help="directory with manifest.json (.bin, same tokenizer) or "
                        "raw_text_manifest.json (jsonl, cross-tokenizer)")
    p.add_argument("--sources", nargs="*", default=None,
                   help="subset of D_x sources (default: all in the manifest)")
    p.add_argument("--posttrain_json", default="",
                   help="safety post-training data (required when alpha < 1)")
    p.add_argument("--posttrain_max_seq_length", type=int, default=defaults.posttrain_max_seq_length)
    p.add_argument("--rollout_prefix_len", type=int, default=defaults.rollout_prefix_len)
    p.add_argument("--rollout_gen_len", type=int, default=defaults.rollout_gen_len)
    p.add_argument("--n_responses", type=int, default=defaults.n_responses)
    p.add_argument("--rollout_temperature", type=float, default=defaults.rollout_temperature)
    p.add_argument("--rollout_top_p", type=float, default=defaults.rollout_top_p)
    p.add_argument("--alpha", type=float, default=defaults.alpha)
    p.add_argument("--top_k", type=int, default=defaults.top_k)
    p.add_argument("--gold_beta", type=float, default=defaults.gold_beta)
    p.add_argument("--gold_matched_weight", type=float, default=None)
    p.add_argument("--gold_unmatched_weight", type=float, default=None)
    p.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    p.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    p.add_argument("--grad_clip", type=float, default=defaults.grad_clip)
    p.add_argument("--warmup_ratio", type=float, default=defaults.warmup_ratio)
    p.add_argument("--global_batch_size", type=int, default=defaults.global_batch_size)
    p.add_argument("--micro_batch_size", type=int, default=defaults.micro_batch_size)
    p.add_argument("--max_steps", type=int, default=defaults.max_steps)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--output_dir", default=defaults.output_dir)
    p.add_argument("--save_every_steps", type=int, default=defaults.save_every_steps)
    p.add_argument("--log_every_steps", type=int, default=defaults.log_every_steps)
    a = p.parse_args()
    if not 0.0 <= a.alpha <= 1.0:
        p.error("--alpha must be in [0, 1]")
    if a.alpha < 1.0 and not a.posttrain_json:
        p.error("--posttrain_json is required when alpha < 1")
    kwargs = vars(a)
    kwargs["gradient_checkpointing"] = not kwargs.pop("no_gradient_checkpointing")
    return DistillConfig(**kwargs)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    trainer = SafetyDistillTrainer(parse_args())
    trainer.train()
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
