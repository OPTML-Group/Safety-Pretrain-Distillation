# Can Native Safety Travel? Reusing Safety Pretraining Across Models via On-Policy Distillation

This repository transfers the safety of a safety-pretrained teacher to a standard-pretrained student through on-policy distillation.

## Method

Each training step optimizes

```
L = alpha * L_distill + (1 - alpha) * L_util
```

- `L_distill` is on-policy distillation. The student samples responses to prompts drawn from the safety-pretraining corpora (SafeWeb, RefuseWeb, Moral Education), and the frozen teacher scores the student-generated tokens.
  - **Same tokenizer:** reverse KL restricted to the student's top-k tokens at each position, with both distributions renormalized over that set (k = 16).
  - **Different tokenizers:** GOLD. The two tokenizations are aligned by byte offsets. Within each aligned group, matched tokens are compared with the Jensen–Shannon divergence and unmatched tokens with the ULD distance.
- `L_util` is the cross-entropy on safety post-training instruction data (UltraChat, WildGuardMix, WildJailbreak, RefuseWeb), formatted with the student's chat template. Training therefore produces a chat model directly, without a separate instruction-tuning stage.

The student trains in fp32. The teacher stays in bf16, and its logits are cast to fp32 before any loss is computed.

## Repository layout

```
safety_distill/
  train.py         trainer and command-line entry point
  losses.py        top-k on-policy distillation loss, masked cross-entropy
  gold.py          GOLD cross-tokenizer loss (adapted from TRL, see NOTICE)
  data.py          prompt and instruction datasets
  chat_format.py   chat templates for base-model students
data/
  prepare_distillation_data.py   distillation prompts (D_x)
  prepare_posttrain_data.py      safety post-training data (D_post)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**1. Prepare data.**

```bash
python data/prepare_distillation_data.py --output_dir data/distillation
python data/prepare_posttrain_data.py --output_dir data/safety_posttrain
```

If the student's tokenizer differs from the teacher's, add `--output_format raw_text` to the first command.

**2. Distill.** The trainer compares the student and teacher vocabularies. If they match, it uses top-k OPD with pre-tokenized `.bin` prompts; otherwise it uses GOLD with raw-text prompts.

```bash
accelerate launch --use_deepspeed --zero_stage 2 --num_processes 4 \
    --mixed_precision no --gradient_accumulation_steps 16 --gradient_clipping 1.0 \
    -m safety_distill.train \
    --teacher_model locuslab/safelm-1.7b \
    --student_model HuggingFaceTB/SmolLM2-1.7B \
    --distill_data_root data/distillation \
    --posttrain_json data/safety_posttrain/train.json \
    --alpha 0.7 --max_steps 3000 --output_dir checkpoints/smollm2_distill
```

Under DeepSpeed, two settings must match the trainer:
- `--gradient_accumulation_steps` must equal `global_batch_size / (micro_batch_size * num_processes)`.
- `--gradient_clipping` must equal `--grad_clip`.

For students with 7B or more parameters, use `--zero_stage 3 --zero3_save_16bit_model true`. Run `python -m safety_distill.train --help` for all options.

## Acknowledgments

The GOLD loss is adapted from [Hugging Face TRL](https://github.com/huggingface/trl) (Apache-2.0). The teacher is the safety-pretrained [SafeLM](https://huggingface.co/locuslab/safelm-1.7b).
