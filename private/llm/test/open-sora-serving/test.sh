PY_SCRIPT=finference.py

export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2

echo "Now CUDA_VISIBLE_DEVICES is set to:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for num_frames in 51; do
    for resolution in "360p" "480p" "720p" "1080p"; do
        echo "num_frames: $num_frames, resolution: $resolution, gpus: $NUM_GPUS"
        python -m torch.distributed.run --standalone --nproc_per_node $NUM_GPUS ${PY_SCRIPT} \
        --num_frames ${num_frames} \
        --resolution ${resolution}
    done
done
