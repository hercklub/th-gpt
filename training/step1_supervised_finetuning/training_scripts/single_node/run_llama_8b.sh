#!/bin/bash
# Training script for LLaMA 3.1 8B - Step 1: Supervised Fine-tuning (Full Fine-tuning)
# This is the DEFAULT/RECOMMENDED training method
# Adapted for RunPod deployment

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_llama_8b_sft
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

# DeepSpeed configuration
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# export NCCL_DEBUG=INFO  # Uncomment for debugging

# Note: Adjust --include localhost:0,1,2,3 based on your GPU availability
# For single GPU training, remove the --include parameter entirely

deepspeed --master_port 12345 main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 2e-5 \
   --weight_decay 0.0 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
