# Can Native Safety Travel? Reusing Safety Pretraining Across Models via On-Policy Distillation

Official code for transferring *native safety*, the safety a model acquires during safety pretraining, from a small safety-pretrained teacher to standard-pretrained students through on-policy distillation. The code supports both same-tokenizer and cross-tokenizer (GOLD) students.

## Installation

```bash
pip install -r requirements.txt
```

## Data

```bash
# Distillation prompts (SafeWeb, RefuseWeb, Moral Education)
python data/prepare_distillation_data.py --output_dir data/distillation
# Add --output_format raw_text for a student whose tokenizer differs from the teacher's.

# Instruction data for the utility loss (UltraChat-200k, WildGuardMix, WildJailbreak)
python data/prepare_posttrain_data.py --output_dir data/safety_posttrain
```

## Training

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

Run `python -m safety_distill.train --help` for all options.

## Acknowledgments

The GOLD loss is adapted from [Hugging Face TRL](https://github.com/huggingface/trl) (see `NOTICE`). The teacher is [SafeLM](https://huggingface.co/locuslab/safelm-1.7b) from [Safety Pretraining](https://arxiv.org/abs/2504.16980).
