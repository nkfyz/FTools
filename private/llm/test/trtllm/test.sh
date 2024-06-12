MODEL_PATH=/workspace/models/01-ai/Yi-1.5-34B-Chat
CONVERTED_MODEL_PATH=./tllm_checkpoint_1gpu_tp1
ENGINE=./tmp/yi/trt_engines/fp16/1-gpu/

for INPUT_LEN in 512 1024; do
    for OUTPUT_LEN in 128 256 512; do
        for BATCH_SIZE in 2 4 8 16; do
                echo "Input length: $INPUT_LEN, Output length: $OUTPUT_LEN, Batch size: $BATCH_SIZE"
                python3 datagen.py --input_size $INPUT_LEN --batch_size $BATCH_SIZE
                mpirun -n 1 --allow-run-as-root python3 run.py \
                    --max_output_len $OUTPUT_LEN \
                    --max_input_length 1024 \
                    --input_file ./npydata/input_${BATCH_SIZE}_${INPUT_LEN}.npy \
                    --engine_dir ${ENGINE}
        done
    done
done