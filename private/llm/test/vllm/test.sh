BACKEND="vllm"
MODEL="/home/node-user/models/01-ai/Yi-1.5-34B-Chat"
N=1
NUM_PROMPTS=10
SEED=1024
MAX_MODEL_LEN=1024
DTYPE="float16"
GPU_MEMORY_UTILIZATION=0.9
DEVICE="cuda"
ENABLE_PREFIX_CACHING="--enable-prefix-caching"
PYTHONPATH=/home/node-user/miniconda3/envs/vllm/lib/python3.9

export CUDA_VISIBLE_DEVICES=0

echo "Now CUDA_VISIBLE_DEVICES is set to:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for TP_SIZE in 1; do
    for INPUT_LEN in 512 1024; do
        for OUTPUT_LEN in 128 256 512; do
            for BATCH_SIZE in 2 4 8 16; do
                    echo "Input length: $INPUT_LEN, Output length: $OUTPUT_LEN, Batch size: $BATCH_SIZE, Tensor parallel size: $TP_SIZE"
                    python vllm_test.py \
                        --input-len $INPUT_LEN \
                        --output-len $OUTPUT_LEN \
                        --model $MODEL \
                        --tensor-parallel-size $TP_SIZE \
                        --batch-size $BATCH_SIZE
            done
        done
    done
done
