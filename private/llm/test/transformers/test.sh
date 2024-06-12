PY_SCRIPT=transformer.py
MODEL=/home/node-user/models/01-ai/Yi-1.5-34B-Chat
NAME=Yi-1.5-34B-Chat

export CUDA_VISIBLE_DEVICES=7
NUM_GPUS=1

echo "Now CUDA_VISIBLE_DEVICES is set to:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for max_seq_length in 512; do
    for max_output_length in 512; do
        for bsz in 2 4 8 16; do
            python ${PY_SCRIPT} \
            --model_name ${NAME} \
            --model_path ${MODEL} \
            --input_len ${max_seq_length} \
            --output_len ${max_output_length} \
            --batch_size ${bsz} \
            --num_gpus ${NUM_GPUS}
        done
    done
done
