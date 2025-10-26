#!/bin/bash
# Training script for LLaMA 3.1 8B - Step 3: RLHF (PPO) Fine-tuning
# Requires both Actor and Critic models from Steps 1 and 2

ACTOR_MODEL_PATH=$1
CRITIC_MODEL_PATH=$2
ACTOR_ZERO_STAGE=$3
CRITIC_ZERO_STAGE=$4
OUTPUT=$5

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_llama_8b_rlhf
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi

# Validate required arguments
if [ "$ACTOR_MODEL_PATH" == "" ] || [ "$CRITIC_MODEL_PATH" == "" ]; then
    echo "Error: ACTOR_MODEL_PATH and CRITIC_MODEL_PATH are required!"
    echo "Usage: $0 <ACTOR_MODEL_PATH> <CRITIC_MODEL_PATH> [ACTOR_ZERO_STAGE] [CRITIC_ZERO_STAGE] [OUTPUT]"
    echo "Example: $0 ./output_llama_8b_sft ./output_llama_8b_reward"
    exit 1
fi

mkdir -p $OUTPUT

# DeepSpeed configuration
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# PPO hyperparameters optimized for LLaMA 3.1 8B
Actor_Lr=5e-7
Critic_Lr=5e-6

deepspeed --master_port 12346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 2 \
   --per_device_mini_train_batch_size 2 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 2 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --enable_hybrid_engine \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_ema \
   --output_dir $OUTPUT \
    &> $OUTPUT/training.log
