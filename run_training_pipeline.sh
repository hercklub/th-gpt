#!/bin/bash
# Complete training pipeline for PokerGPT with LLaMA 3.1 8B
# This script runs all three training steps sequentially

set -e  # Exit on error

echo "================================================"
echo "PokerGPT Training Pipeline - LLaMA 3.1 8B"
echo "================================================"

# Configuration
BASE_DIR=$(pwd)
ZERO_STAGE=2  # ZeRO optimization stage (0, 1, 2, or 3)

# Output directories
STEP1_OUTPUT="${BASE_DIR}/training/step1_supervised_finetuning/output_llama_8b_sft"
STEP2_OUTPUT="${BASE_DIR}/training/step2_reward_model_finetuning/output_llama_8b_reward"
STEP3_OUTPUT="${BASE_DIR}/training/step3_rlhf_finetuning/output_llama_8b_rlhf"

# Step selection
RUN_STEP1=${RUN_STEP1:-true}
RUN_STEP2=${RUN_STEP2:-true}
RUN_STEP3=${RUN_STEP3:-true}

# Check for existing models to resume training
if [ -d "$STEP1_OUTPUT" ] && [ "$RUN_STEP1" = false ]; then
    echo "Found existing Step 1 model at $STEP1_OUTPUT"
    echo "Skipping Step 1 training..."
else
    if [ "$RUN_STEP1" = true ]; then
        echo ""
        echo "================================================"
        echo "Step 1: Supervised Fine-tuning (SFT) - Full Fine-tuning"
        echo "================================================"
        cd "${BASE_DIR}/training/step1_supervised_finetuning"
        bash training_scripts/single_node/run_llama_8b.sh $STEP1_OUTPUT $ZERO_STAGE
        echo "Step 1 complete! Model saved to: $STEP1_OUTPUT"
    fi
fi

if [ -d "$STEP2_OUTPUT" ] && [ "$RUN_STEP2" = false ]; then
    echo "Found existing Step 2 model at $STEP2_OUTPUT"
    echo "Skipping Step 2 training..."
else
    if [ "$RUN_STEP2" = true ]; then
        echo ""
        echo "================================================"
        echo "Step 2: Reward Model Fine-tuning"
        echo "================================================"
        cd "${BASE_DIR}/training/step2_reward_model_finetuning"
        bash training_scripts/single_node/run_llama_8b.sh $STEP2_OUTPUT $ZERO_STAGE
        echo "Step 2 complete! Model saved to: $STEP2_OUTPUT"
    fi
fi

if [ "$RUN_STEP3" = true ]; then
    echo ""
    echo "================================================"
    echo "Step 3: RLHF (PPO) Fine-tuning"
    echo "================================================"

    # Verify required models exist
    if [ ! -d "$STEP1_OUTPUT" ]; then
        echo "Error: Actor model not found at $STEP1_OUTPUT"
        echo "Please run Step 1 first or set STEP1_OUTPUT to correct path"
        exit 1
    fi

    if [ ! -d "$STEP2_OUTPUT" ]; then
        echo "Error: Critic model not found at $STEP2_OUTPUT"
        echo "Please run Step 2 first or set STEP2_OUTPUT to correct path"
        exit 1
    fi

    cd "${BASE_DIR}/training/step3_rlhf_finetuning"
    bash training_scripts/single_node/run_llama_8b.sh \
        $STEP1_OUTPUT \
        $STEP2_OUTPUT \
        $ZERO_STAGE \
        $ZERO_STAGE \
        $STEP3_OUTPUT
    echo "Step 3 complete! Model saved to: $STEP3_OUTPUT"
fi

echo ""
echo "================================================"
echo "Training Pipeline Complete!"
echo "================================================"
echo "Final model location: $STEP3_OUTPUT"
echo ""
echo "To evaluate your model, run:"
echo "  python evaluate_model.py --model_path $STEP3_OUTPUT"
