# PokerGPT Training with LLaMA 3.1 8B - Quick Start Guide

This guide provides complete instructions for training a PokerGPT model using LLaMA 3.1 8B on RunPod.io with DeepSpeed optimization.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [RunPod Setup](#runpod-setup)
- [Environment Setup](#environment-setup)
- [Training Pipeline](#training-pipeline)
  - [Step 1: Supervised Fine-tuning (SFT)](#step-1-supervised-fine-tuning-sft)
  - [Step 2: Reward Model Fine-tuning](#step-2-reward-model-fine-tuning)
  - [Step 3: RLHF (PPO) Fine-tuning](#step-3-rlhf-ppo-fine-tuning)
- [Model Evaluation](#model-evaluation)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [Resource Requirements](#resource-requirements)

---

## Overview

This repository implements a three-step training pipeline for PokerGPT:

1. **Supervised Fine-tuning (SFT)**: Train the base LLaMA 3.1 8B model on poker gameplay data
2. **Reward Model Fine-tuning**: Train a reward model to evaluate poker decisions
3. **RLHF/PPO Fine-tuning**: Use Proximal Policy Optimization to align the model with optimal poker strategy

The training uses DeepSpeed for efficient distributed training and supports LoRA for memory-efficient fine-tuning.

---

## Prerequisites

### Required Accounts

1. **HuggingFace Account**:
   - Sign up at https://huggingface.co/join
   - Accept LLaMA 3.1 license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
   - Create an access token at https://huggingface.co/settings/tokens (with "Read" permissions)

2. **RunPod Account**:
   - Sign up at https://runpod.io
   - Add credits to your account (recommended: $20-50 for initial experiments)

### Hardware Requirements

**Recommended RunPod Pod Configuration:**
- **GPU**: 1x A40 (48GB) or 1x A100 (40GB/80GB)
- **RAM**: 64GB+
- **Storage**: 200GB+ NVMe SSD
- **Template**: PyTorch 2.0+ with CUDA 11.8+

**Minimum Configuration (with LoRA):**
- GPU: 1x RTX 4090 (24GB) or 1x A10 (24GB)
- RAM: 32GB+
- Storage: 100GB+

---

## RunPod Setup

### 1. Create a RunPod Instance

1. Log in to RunPod.io
2. Click **"Deploy"** → **"GPU Pods"**
3. Select a GPU instance (recommended: A40 48GB)
4. Choose a PyTorch template (e.g., "RunPod PyTorch 2.1")
5. Set disk size to at least 200GB
6. Click **"Deploy On-Demand"** or **"Deploy Spot"** (spot is cheaper but can be interrupted)

### 2. Connect to Your Pod

Once your pod is running:

```bash
# Click "Connect" on your pod
# Copy the SSH command, it looks like:
ssh root@<pod-id>.ssh.runpod.io -p <port> -i ~/.ssh/id_ed25519

# Or use the web terminal in RunPod interface
```

### 3. Clone the Repository

```bash
cd /workspace  # RunPod's persistent storage
git clone <your-repo-url>
cd <repo-name>
```

---

## Environment Setup

### Quick Setup (Automated)

Run the automated setup script:

```bash
bash setup_runpod.sh
```

This script will:
- Install all dependencies (PyTorch, DeepSpeed, Transformers, etc.)
- Set up HuggingFace authentication
- Create necessary directories
- Verify the installation

**During setup, you'll be prompted for your HuggingFace token. Have it ready!**

### Manual Setup (Alternative)

If you prefer manual installation:

```bash
# Update pip
pip install --upgrade pip

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

### Verify Installation

```bash
# Test DeepSpeed
ds_report

# Test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Test HuggingFace access
python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B'); print('HF access OK')"
```

---

## Training Pipeline

### Complete Pipeline (Automated)

To run all three training steps sequentially:

```bash
bash run_training_pipeline.sh
```

This will:
1. Train SFT model (Step 1)
2. Train Reward model (Step 2)
3. Train RLHF model (Step 3)

**Note**: Full training can take 6-24 hours depending on GPU and dataset size.

### Manual Step-by-Step Training

For more control, run each step individually:

---

### Step 1: Supervised Fine-tuning (SFT)

Train the base LLaMA 3.1 8B model on poker gameplay data.

#### Using LoRA (Recommended for limited GPU memory)

```bash
cd training/step1_supervised_finetuning

# Run with LoRA (memory efficient)
bash training_scripts/single_node/run_llama_8b_lora.sh \
    ./output_llama_8b_sft_lora \
    2  # ZeRO stage
```

**LoRA Training Parameters:**
- Batch size: 4 per device
- Gradient accumulation: 2 steps
- Epochs: 3
- Learning rate: 5e-4
- LoRA rank: 128

#### Full Fine-tuning (Requires more GPU memory)

```bash
cd training/step1_supervised_finetuning

# Run full fine-tuning
bash training_scripts/single_node/run_llama_8b.sh \
    ./output_llama_8b_sft \
    2  # ZeRO stage
```

**Full Training Parameters:**
- Batch size: 2 per device
- Gradient accumulation: 4 steps
- Epochs: 3
- Learning rate: 2e-5

#### Monitor Training

```bash
# View training logs in real-time
tail -f output_llama_8b_sft_lora/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

#### Expected Output

Training will create:
- `output_llama_8b_sft_lora/pytorch_model.bin` - Model weights
- `output_llama_8b_sft_lora/config.json` - Model configuration
- `output_llama_8b_sft_lora/training.log` - Training logs

**Training Time**: 4-8 hours on A40 (depends on dataset size)

---

### Step 2: Reward Model Fine-tuning

Train a reward model to evaluate poker decisions.

```bash
cd training/step2_reward_model_finetuning

# Train reward model
bash training_scripts/single_node/run_llama_8b.sh \
    ./output_llama_8b_reward \
    2  # ZeRO stage
```

**Reward Model Parameters:**
- Uses paired good/bad responses
- Training epochs: 1 (to avoid overfitting)
- Batch size: 2 per device
- Learning rate: 5e-6
- Dropout: Disabled

#### Monitor Training

```bash
tail -f output_llama_8b_reward/training.log
```

#### Expected Metrics

Look for in the logs:
- **Accuracy**: Percentage where good responses score higher than bad responses (target: >60%)
- **Reward scores**: Higher scores for good responses vs bad responses

**Training Time**: 2-4 hours on A40

---

### Step 3: RLHF (PPO) Fine-tuning

Use Proximal Policy Optimization to align the model with optimal poker strategy.

**Prerequisites**: Both Step 1 and Step 2 models must be trained.

```bash
cd training/step3_rlhf_finetuning

# Train RLHF model
bash training_scripts/single_node/run_llama_8b.sh \
    ../step1_supervised_finetuning/output_llama_8b_sft_lora \  # Actor model
    ../step2_reward_model_finetuning/output_llama_8b_reward \  # Critic model
    2 \  # Actor ZeRO stage
    2 \  # Critic ZeRO stage
    ./output_llama_8b_rlhf  # Output directory
```

**RLHF Parameters:**
- Uses both Actor (from Step 1) and Critic (from Step 2)
- PPO epochs: 1
- Generation batch: 1
- Actor LR: 5e-7
- Critic LR: 5e-6
- Enables EMA (Exponential Moving Average)

#### Monitor Training

```bash
tail -f output_llama_8b_rlhf/training.log

# Watch reward scores - they should generally increase over time
```

**Training Time**: 6-12 hours on A40

**Note**: RLHF training can be unstable. If you see divergence (NaN losses), try:
- Reducing learning rates (actor_lr and critic_lr)
- Using gradient clipping
- Reducing batch size

---

## Model Evaluation

### Interactive Evaluation

Test your trained model interactively:

```bash
python evaluate_model.py \
    --model_path training/step3_rlhf_finetuning/output_llama_8b_rlhf \
    --mode interactive
```

Example prompts:
```
Your prompt: What should I do with pocket aces in early position?

Your prompt: Hero has KK on board AK347. Villain bets pot. What should hero do?
```

### Batch Evaluation

Test on multiple prompts from a file:

```bash
# Create test file
cat > test_prompts.txt << EOF
What is the best starting hand in Texas Hold'em?
Should I call or fold with AK suited pre-flop?
How do I play middle pair on the flop?
EOF

# Run batch evaluation
python evaluate_model.py \
    --model_path training/step3_rlhf_finetuning/output_llama_8b_rlhf \
    --mode batch \
    --test_file test_prompts.txt
```

Results will be saved to `test_prompts_results.txt`.

### Evaluation Options

```bash
python evaluate_model.py \
    --model_path <path> \
    --mode interactive \
    --temperature 0.7 \      # Sampling temperature (0.0-1.0)
    --max_length 512 \       # Maximum generation length
    --device cuda            # cuda or cpu
```

---

## Configuration Options

### Adjusting Training Parameters

Edit the training scripts to customize:

#### Batch Size and Memory

```bash
# In training scripts (.sh files), adjust:
--per_device_train_batch_size 4    # Increase/decrease based on GPU memory
--gradient_accumulation_steps 2    # Increase to simulate larger batch
```

**Memory Guidelines:**
- 24GB GPU: batch_size=1, grad_accum=8
- 40GB GPU: batch_size=2, grad_accum=4
- 48GB GPU: batch_size=4, grad_accum=2
- 80GB GPU: batch_size=8, grad_accum=1

#### Learning Rates

```bash
# Step 1 (SFT)
--learning_rate 2e-5     # Lower = more stable, Higher = faster convergence

# Step 2 (Reward)
--learning_rate 5e-6     # Keep relatively low to avoid overfitting

# Step 3 (RLHF)
--actor_learning_rate 5e-7
--critic_learning_rate 5e-6
```

#### Training Epochs

```bash
--num_train_epochs 3     # More epochs = more training, risk of overfitting
```

### Using Custom Data

To use your own poker training data:

1. **Step 1 (SFT)**: Format as instruction-following dataset
   ```json
   {
     "prompt": "Hero has AK in position...",
     "response": "Hero should raise to 3x..."
   }
   ```

2. **Step 2 (Reward)**: Format as paired comparisons
   ```json
   {
     "prompt": "Hero has AK...",
     "chosen": "Raise to 3x",
     "rejected": "Fold"
   }
   ```

3. Update `--data_path` in training scripts to point to your data

### ZeRO Optimization Stages

DeepSpeed ZeRO stages trade-off memory vs speed:

- **Stage 0**: No optimization (fastest, most memory)
- **Stage 1**: Optimizer state partitioning
- **Stage 2**: Gradient partitioning (recommended)
- **Stage 3**: Parameter partitioning (slowest, least memory)

Change via `--zero_stage` parameter or `$ZERO_STAGE` variable.

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `--per_device_train_batch_size 1`
- Increase gradient accumulation: `--gradient_accumulation_steps 8`
- Use LoRA training (Step 1)
- Increase ZeRO stage: `--zero_stage 3`
- Enable gradient checkpointing (already enabled in scripts)

#### 2. HuggingFace Authentication Error

**Error**: `Repository not found` or `Access denied`

**Solutions**:
```bash
# Re-login to HuggingFace
huggingface-cli login
# Enter your token

# Verify you accepted the license
# Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
# Click "Accept" on the license agreement
```

#### 3. DeepSpeed Installation Issues

**Error**: `No module named 'deepspeed'`

**Solutions**:
```bash
# Install with pip
pip install deepspeed

# Or build from source (if pip fails)
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
pip install .
```

#### 4. NCCL Errors in Multi-GPU Training

**Error**: `NCCL initialization failed`

**Solutions**:
```bash
# Already set in scripts, but verify:
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO  # For debugging
```

#### 5. Training Divergence (NaN losses)

**Symptoms**: Loss becomes NaN, model outputs gibberish

**Solutions**:
- Reduce learning rates (especially in Step 3)
- Check data quality (no corrupted examples)
- Reduce batch size
- Enable gradient clipping (add `--max_grad_norm 1.0` to scripts)

#### 6. Slow Training Speed

**Solutions**:
- Use LoRA instead of full fine-tuning
- Increase batch size if memory allows
- Use lower ZeRO stage (0 or 1)
- Enable flash attention (requires compatible GPU)
- Use fp16/bf16 mixed precision (already enabled via DeepSpeed)

### Monitoring System Resources

```bash
# GPU usage
watch -n 1 nvidia-smi

# CPU and memory
htop

# Disk usage
df -h
du -sh training/step*/output*

# Check training progress
tail -f training/step1_supervised_finetuning/output*/training.log
```

---

## Resource Requirements

### Estimated Costs on RunPod (Spot Pricing)

| Step | GPU | Duration | Cost (approx) |
|------|-----|----------|---------------|
| Step 1 (SFT) | A40 48GB | 4-8 hours | $2.40 - $4.80 |
| Step 2 (Reward) | A40 48GB | 2-4 hours | $1.20 - $2.40 |
| Step 3 (RLHF) | A40 48GB | 6-12 hours | $3.60 - $7.20 |
| **Total** | | **12-24 hours** | **$7.20 - $14.40** |

*Note: Prices vary based on availability and region. On-demand pricing is typically 2-3x higher.*

### Storage Requirements

- Base model download: ~16 GB
- Step 1 output: ~16 GB (8GB with LoRA)
- Step 2 output: ~16 GB
- Step 3 output: ~16 GB
- Logs and cache: ~5 GB
- **Total**: ~70 GB (recommended 200GB for safety)

### Training Time Estimates

Based on A40 48GB GPU:

| Dataset Size | Step 1 | Step 2 | Step 3 | Total |
|--------------|--------|--------|--------|-------|
| Small (10K examples) | 2h | 1h | 3h | 6h |
| Medium (50K examples) | 6h | 3h | 8h | 17h |
| Large (200K examples) | 16h | 8h | 20h | 44h |

---

## Advanced Tips

### Using Weights & Biases (W&B) for Tracking

```bash
# Install wandb
pip install wandb
wandb login

# Add to training scripts
--report_to wandb \
--run_name "poker_llama_sft"
```

### Multi-GPU Training

For multiple GPUs, modify the DeepSpeed command:

```bash
# In training scripts, change:
deepspeed --master_port 12345 main.py \
    # ... parameters ...

# To:
deepspeed --num_gpus 4 --master_port 12345 main.py \
    # ... parameters ...

# Or specify GPU IDs:
deepspeed --include localhost:0,1,2,3 --master_port 12345 main.py \
    # ... parameters ...
```

### Resuming from Checkpoint

To resume interrupted training:

```bash
# DeepSpeed automatically saves checkpoints
# Just re-run the same command with the same output directory
bash training_scripts/single_node/run_llama_8b.sh ./output_llama_8b_sft 2

# DeepSpeed will detect existing checkpoints and resume
```

### Exporting for Deployment

After training, export the final model:

```bash
# The model is already in HuggingFace format
# You can deploy it directly from the output directory

# Or push to HuggingFace Hub
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('./training/step3_rlhf_finetuning/output_llama_8b_rlhf')
model.push_to_hub('your-username/poker-llama-8b')
"
```

---

## Next Steps

After completing training:

1. **Evaluate thoroughly**: Test on diverse poker scenarios
2. **Fine-tune hyperparameters**: Adjust learning rates, batch sizes based on results
3. **Gather more data**: More training data generally improves performance
4. **Deploy**: Set up inference API or integrate into your application
5. **Monitor in production**: Track model performance on real queries

## Support and Resources

- **DeepSpeed Documentation**: https://www.deepspeed.ai/
- **Transformers Documentation**: https://huggingface.co/docs/transformers
- **LLaMA 3.1 Model Card**: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
- **RunPod Documentation**: https://docs.runpod.io/

## Contributing

Found issues or improvements? Please open an issue or pull request!

---

**Happy Training! 🃏🤖**
