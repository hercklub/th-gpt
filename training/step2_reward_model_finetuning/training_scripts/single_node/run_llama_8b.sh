#!/bin/bash
# Training script for LLaMA 3.1 8B - Step 2: Reward Model Fine-tuning

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_llama_8b_reward
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
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 512 \
   --learning_rate 5e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --disable_dropout \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
