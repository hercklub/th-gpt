# Training Cheat Sheet - Quick Command Reference

## Initial Setup

```bash
# 1. Clone repository
git clone <your-repo-url>
cd <repo-name>

# 2. Run setup
bash setup_runpod.sh

# 3. Login to HuggingFace
huggingface-cli login
```

## Training Commands

### Full Pipeline (All 3 Steps)
```bash
bash run_training_pipeline.sh
```

### Individual Steps

#### Step 1: Supervised Fine-tuning (SFT)
```bash
cd training/step1_supervised_finetuning

# With LoRA (memory efficient)
bash training_scripts/single_node/run_llama_8b_lora.sh ./output_llama_8b_sft_lora 2

# Full fine-tuning
bash training_scripts/single_node/run_llama_8b.sh ./output_llama_8b_sft 2
```

#### Step 2: Reward Model
```bash
cd training/step2_reward_model_finetuning
bash training_scripts/single_node/run_llama_8b.sh ./output_llama_8b_reward 2
```

#### Step 3: RLHF/PPO
```bash
cd training/step3_rlhf_finetuning
bash training_scripts/single_node/run_llama_8b.sh \
    ../step1_supervised_finetuning/output_llama_8b_sft_lora \
    ../step2_reward_model_finetuning/output_llama_8b_reward \
    2 2 ./output_llama_8b_rlhf
```

## Monitoring

```bash
# Watch training logs
tail -f <output_directory>/training.log

# Monitor GPU
watch -n 1 nvidia-smi

# Check disk space
df -h
```

## Evaluation

```bash
# Interactive mode
python evaluate_model.py \
    --model_path training/step3_rlhf_finetuning/output_llama_8b_rlhf \
    --mode interactive

# Batch mode
python evaluate_model.py \
    --model_path training/step3_rlhf_finetuning/output_llama_8b_rlhf \
    --mode batch \
    --test_file test_prompts.txt
```

## Troubleshooting Quick Fixes

```bash
# Out of memory → Reduce batch size
--per_device_train_batch_size 1 --gradient_accumulation_steps 8

# HuggingFace auth error
huggingface-cli login

# Verify setup
ds_report
python -c "import torch; print(torch.cuda.is_available())"
```

## Key Parameters to Adjust

| Parameter | Location | Description |
|-----------|----------|-------------|
| `--per_device_train_batch_size` | Training scripts | Batch size per GPU |
| `--gradient_accumulation_steps` | Training scripts | Accumulate gradients |
| `--learning_rate` | Training scripts | Learning rate |
| `--num_train_epochs` | Training scripts | Training epochs |
| `--lora_dim` | Step 1 LoRA script | LoRA rank |
| `--zero_stage` | Command line arg | DeepSpeed ZeRO stage |

## File Locations

- Training scripts: `training/step*_*/training_scripts/single_node/`
- Output models: `training/step*_*/output_*/`
- Training logs: `training/step*_*/output_*/training.log`
- Main training code: `training/step*_*/main.py`
- Utilities: `training/utils/`

## Model Paths

- **Base model**: `meta-llama/Meta-Llama-3.1-8B`
- **Step 1 output**: `training/step1_supervised_finetuning/output_llama_8b_sft_lora/`
- **Step 2 output**: `training/step2_reward_model_finetuning/output_llama_8b_reward/`
- **Step 3 output**: `training/step3_rlhf_finetuning/output_llama_8b_rlhf/`

## Environment Variables

```bash
# Disable InfiniBand (for cloud environments)
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Debug NCCL issues
export NCCL_DEBUG=INFO

# Set HuggingFace cache
export HF_HOME=/workspace/.cache/huggingface
```

## Quick Tests

```bash
# Test imports
python -c "import torch, deepspeed, transformers; print('All imports OK')"

# Test GPU
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Test HuggingFace
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B'); print('HF OK')"

# Test DeepSpeed
ds_report
```

For detailed information, see [QUICK_START.md](QUICK_START.md)
