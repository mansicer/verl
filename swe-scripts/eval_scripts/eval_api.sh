export OPENAI_API_KEY=gpt-7698b9529be85067205fa6c29d7f

BASE_URL=https://gpt-bj.singularity-ai.com/gpt-proxy
MODEL_NAME=deepseek-chat
TEMPERATURE=0.0
MAX_TOKENS=4096
MAX_MODEL_LEN=131072

DATASET_PATH=data/swe-verified-eval/search-replace.parquet
OUTPUT_PATH=outputs/swe-verified-eval/search-replace/deepseek-chat-temp-${TEMPERATURE}-tokens-${MAX_TOKENS}-maxlen-${MAX_MODEL_LEN}.jsonl


python swe-scripts/eval_scripts/generate_rollout_api.py \
    --dataset $DATASET_PATH \
    --output_path $OUTPUT_PATH \
    --base_url $BASE_URL \
    --model_name $MODEL_NAME \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --model_max_length $MAX_MODEL_LEN 
