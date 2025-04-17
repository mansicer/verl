WORKING_DIR=/mnt/data/fuxiang/verl-swe
cd $WORKING_DIR

DATA_NAME=swe-oracle-extra
TRAIN_DATA_PATH="[$(echo ${WORKING_DIR}/data/swe-oracle/search-replace/train-16k.parquet),$(echo ${WORKING_DIR}/data/swe-bench-extra/search-replace/train-16k.parquet),$(echo ${WORKING_DIR}/data/swe-gym/search_replace/train-16k.parquet)]"
TEST_DATA_PATH=${WORKING_DIR}/data/swe-oracle/search-replace/test-16k.parquet

MODEL_NAME=Qwen2.5-Coder-32B-Instruct
MODEL_PATH=/mnt/data/models/Qwen2.5-Coder-32B-Instruct

REWARD_FN_PATH=${WORKING_DIR}/verl/utils/reward_score/swe_rl/search_replace.py

EXP_NOTE=search-replace-16k
GPU_NUMS=8

PROJECT_NAME=swe-rl-exp
EXPERIMENT_NAME=${EXP_NOTE}-${MODEL_NAME}-${DATA_NAME}-NODE${WORLD_SIZE}
OUTPUT_DIR=${WORKING_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

PROMPT_LENGTH=12288
RESPONSE_LENGTH=4096

TP_SIZE=4
GPU_MEMORY_UTILIZATION=0.80
SP_SIZE=1
NUM_BATCHED_TOKENS=$((${PROMPT_LENGTH} + ${RESPONSE_LENGTH}))
MAX_TOKEN_PER_GPU=$(((${PROMPT_LENGTH} + ${RESPONSE_LENGTH}) / ${SP_SIZE}))

unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE
export WANDB_API_KEY=f33968783cdb63b751908c83a60d52b89d3c8e51
export WANDB_ENTITY=skywork

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${TEST_DATA_PATH} \
    data.train_batch_size=128 \
    data.max_prompt_length=${PROMPT_LENGTH} \
    data.max_response_length=${RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=128 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=naive \
    custom_reward_function.path=${REWARD_FN_PATH} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.wandb_entity=${WANDB_ENTITY} \
    trainer.log_val_generations=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${GPU_NUMS} \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.rejection_sample=True \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.resume_mode=auto \
    trainer.total_epochs=10

python swe-scripts/model_utils/convert_final_ckpt.py ${OUTPUT_DIR}
