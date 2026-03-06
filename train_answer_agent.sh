#!/bin/bash
# Training script for the Answer Agent (Stage 2)
# Faithful to Memory-R1 paper: trains AA with GRPO using EM reward.
# Memory Manager is frozen; memory banks were built by the trained MM.

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='data/memory_r1_train'

WANDB_PROJECT='Memory-R1-AnswerAgent'

# ---- Model selection ----
export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=memory-r1-answer-grpo-qwen2.5-3b-it
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=memory-r1-answer-grpo-llama3.1-8b-it

export VLLM_ATTENTION_BACKEND=XFORMERS

# ---- Prerequisites ----
# 1. Train Memory Manager first (train_memory_manager.sh)
# 2. Rebuild memory banks using trained MM:
#    python -m memory_r1.inference --mode rebuild_banks \
#        --mm_checkpoint verl_checkpoints/<MM_EXPERIMENT>/... \
#        --locomo_path data/locomo/locomo.json \
#        --output_dir $DATA_DIR
# 3. Re-generate AA training data with updated banks:
#    python -m memory_r1.data.data_construction \
#        --locomo_path data/locomo/locomo.json \
#        --output_dir $DATA_DIR --use_gpt

# ---- Train Answer Agent with GRPO ----
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_answer_agent \
    data.train_files=$DATA_DIR/aa_train.parquet \
    data.val_files=$DATA_DIR/aa_test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=128 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.max_start_length=2048 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=false \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=1 \
    2>&1 | tee $EXPERIMENT_NAME.log
