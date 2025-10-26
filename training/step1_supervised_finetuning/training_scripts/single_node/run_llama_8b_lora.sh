#!/bin/bash
# Training script for LLaMA 3.1 8B with LoRA - Step 1: Supervised Fine-tuning
# LoRA enables efficient training with reduced memory requirements
# Recommended for RunPod with limited GPU memory

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_llama_8b_sft_lora
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

# DeepSpeed configuration
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

deepspeed --master_port 12345 main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 5e-4 \
   --weight_decay 0.0 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --lora_dim 128 \
   --lora_module_name model.layers. \
   --only_optimize_lora \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
