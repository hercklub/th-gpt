#!/bin/bash
# RunPod Environment Setup Script for PokerGPT Training with LLaMA 3.1 8B
# This script prepares the RunPod environment for training

set -e  # Exit on error

echo "================================================"
echo "Setting up RunPod environment for PokerGPT Training"
echo "================================================"

# Update package manager
echo "Updating system packages..."
apt-get update -qq

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y -qq git wget curl build-essential

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust CUDA version as needed)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DeepSpeed
echo "Installing DeepSpeed..."
pip install deepspeed

# Install Transformers and related libraries
echo "Installing Transformers and HuggingFace libraries..."
pip install transformers>=4.35.0 accelerate datasets evaluate

# Install additional dependencies
echo "Installing additional dependencies..."
pip install sentencepiece protobuf wandb tensorboard

# Install ninja for faster compilation
pip install ninja

# Setup HuggingFace authentication
echo ""
echo "================================================"
echo "HuggingFace Authentication Setup"
echo "================================================"
echo "To access LLaMA 3.1 8B, you need to:"
echo "1. Accept the license at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B"
echo "2. Generate a token at: https://huggingface.co/settings/tokens"
echo ""
read -p "Enter your HuggingFace token (or press Enter to skip): " hf_token

if [ ! -z "$hf_token" ]; then
    pip install huggingface_hub
    huggingface-cli login --token $hf_token
    echo "HuggingFace authentication successful!"
else
    echo "Skipping HuggingFace authentication. You can set it later with:"
    echo "  huggingface-cli login"
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p training/step1_supervised_finetuning/output
mkdir -p training/step2_reward_model_finetuning/output
mkdir -p training/step3_rlhf_finetuning/output
mkdir -p logs

# Make training scripts executable
echo "Making training scripts executable..."
chmod +x training/step1_supervised_finetuning/training_scripts/single_node/*.sh
chmod +x training/step2_reward_model_finetuning/training_scripts/single_node/*.sh
chmod +x training/step3_rlhf_finetuning/training_scripts/single_node/*.sh

# Test DeepSpeed installation
echo ""
echo "Testing DeepSpeed installation..."
ds_report

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo "Next steps:"
echo "1. Review and customize training scripts in training/step*_*/training_scripts/"
echo "2. Prepare your poker training data"
echo "3. Start training with Step 1 (SFT)"
echo ""
echo "For detailed instructions, see QUICK_START.md"
