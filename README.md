# PokerGPT Training with LLaMA 3.1 8B

This repository provides a complete training pipeline for PokerGPT using **LLaMA 3.1 8B** with DeepSpeed optimization. It implements a three-step RLHF training process optimized for RunPod.io deployment.

## 🚀 Quick Start

**New to the project? Start here:**

1. **[QUICK_START.md](QUICK_START.md)** - Complete step-by-step training guide
2. **[TRAINING_CHEATSHEET.md](TRAINING_CHEATSHEET.md)** - Quick command reference

### One-Command Setup

```bash
# On RunPod or any CUDA-enabled machine
bash setup_runpod.sh
```

### One-Command Training

```bash
# Train all 3 steps
bash run_training_pipeline.sh
```

## 📋 What's New - LLaMA 3.1 8B Update

This repository is based on DeepSpeed-Chat and has been updated to support:

- ✅ **LLaMA 3.1 8B** (upgraded from OPT-1.3B)
- ✅ **RunPod.io** deployment ready
- ✅ **LoRA training** for memory efficiency
- ✅ **Automated setup** scripts
- ✅ **Interactive evaluation** tools
- ✅ **Comprehensive documentation**

### Training Pipeline

The training consists of 3 steps:

1. **Step 1: Supervised Fine-tuning (SFT)**
   - Train LLaMA 3.1 8B on poker gameplay data
   - Supports LoRA for efficient training

2. **Step 2: Reward Model Fine-tuning**
   - Train a reward model to evaluate poker decisions
   - Uses paired good/bad response comparisons

3. **Step 3: RLHF (PPO) Fine-tuning**
   - Use Proximal Policy Optimization
   - Align model with optimal poker strategy

## 📁 Repository Structure

```
.
├── training/
│   ├── step1_supervised_finetuning/
│   │   ├── main.py
│   │   └── training_scripts/single_node/
│   │       ├── run_llama_8b.sh           # Full fine-tuning
│   │       └── run_llama_8b_lora.sh      # LoRA training (recommended)
│   ├── step2_reward_model_finetuning/
│   │   ├── main.py
│   │   └── training_scripts/single_node/
│   │       └── run_llama_8b.sh
│   ├── step3_rlhf_finetuning/
│   │   ├── main.py
│   │   └── training_scripts/single_node/
│   │       └── run_llama_8b.sh
│   └── utils/                             # Shared utilities
│       ├── model/
│       ├── data/
│       └── utils.py                       # Updated tokenizer handling
├── local_data/                            # Data processing scripts
│   ├── unzip_script.py
│   ├── order_change.py
│   ├── good_players.py
│   └── prompt_engineering3.py
├── setup_runpod.sh                        # Automated environment setup
├── run_training_pipeline.sh               # Run all 3 steps
├── evaluate_model.py                      # Model evaluation tool
├── requirements.txt                       # Python dependencies
├── QUICK_START.md                         # Detailed training guide
└── TRAINING_CHEATSHEET.md                # Quick command reference
```

## 🔧 Requirements

- **GPU**: Minimum 24GB VRAM (40GB+ recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 200GB+ free space
- **Python**: 3.8+
- **CUDA**: 11.8+

See [QUICK_START.md](QUICK_START.md#resource-requirements) for detailed requirements.

## 📊 Training Overview

| Step | Model | Duration (A40) | GPU Memory |
|------|-------|----------------|------------|
| Step 1 (SFT LoRA) | LLaMA 3.1 8B | 4-8 hours | ~30GB |
| Step 2 (Reward) | LLaMA 3.1 8B | 2-4 hours | ~35GB |
| Step 3 (RLHF) | Both models | 6-12 hours | ~40GB |

## 🎯 Key Features

- **Memory Efficient**: LoRA training reduces memory requirements by 50%
- **DeepSpeed ZeRO**: Distributed training optimization
- **Gradient Checkpointing**: Enabled by default for memory savings
- **Mixed Precision**: FP16 training for faster convergence
- **Flexible Configuration**: Easy-to-modify training scripts
- **Comprehensive Logging**: Track training progress in real-time

## 📖 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Complete training guide with:
  - RunPod setup instructions
  - Step-by-step training walkthrough
  - Configuration options
  - Troubleshooting guide
  - Resource requirements

- **[TRAINING_CHEATSHEET.md](TRAINING_CHEATSHEET.md)** - Quick reference for:
  - Common commands
  - Key parameters
  - File locations
  - Quick fixes

## 🔍 Model Evaluation

After training, evaluate your model:

```bash
# Interactive evaluation
python evaluate_model.py \
    --model_path training/step3_rlhf_finetuning/output_llama_8b_rlhf \
    --mode interactive

# Batch evaluation
python evaluate_model.py \
    --model_path training/step3_rlhf_finetuning/output_llama_8b_rlhf \
    --mode batch \
    --test_file test_prompts.txt
```

## 🎲 Data Processing

Original data processing pipeline (in `local_data/`):

1. **Unzip datasets**: `python unzip_script.py`
2. **Filter showdown games**: `python order_change.py`
3. **Filter good players**: `python good_players.py`
4. **Prompt engineering**: `python prompt_engineering3.py`

Final prompt files start with `prompt_*`.

## 💰 Cost Estimates (RunPod Spot)

- **Full training pipeline**: $7-15 on A40 (48GB)
- **Single step testing**: $2-5
- **Total storage**: ~70GB (200GB recommended)

See [QUICK_START.md](QUICK_START.md#resource-requirements) for detailed pricing.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

Based on DeepSpeed-Chat. See original repository for license details.

## 🙏 Acknowledgments

- **DeepSpeed-Chat**: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
- **Meta LLaMA**: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
- **HuggingFace Transformers**: https://github.com/huggingface/transformers

## 📞 Support

For issues or questions:
1. Check [QUICK_START.md](QUICK_START.md#troubleshooting)
2. Review [TRAINING_CHEATSHEET.md](TRAINING_CHEATSHEET.md)
3. Open an issue on GitHub

---

**Ready to train? Start with [QUICK_START.md](QUICK_START.md)!** 🃏🤖
