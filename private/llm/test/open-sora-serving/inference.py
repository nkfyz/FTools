import torch
from colossalai.utils import get_current_device

from opensora_serving.configs import GenerationConfig, LoadConfig
from opensora_serving.configs_preset import _PRESET_MODULE_CONFIG_REGISTRY
from opensora_serving.engine import Engine

import argparse, time

GIGABYTE = 1024**3


def print_memory_usage(prefix: str, device: torch.device, latency, num_frames, resolution):
    torch.cuda.synchronize()
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    max_memory_reserved = torch.cuda.max_memory_reserved(device)
    
    mem1 = max_memory_allocated / GIGABYTE
    mem2 = max_memory_reserved / GIGABYTE
    
    # print(f"{prefix}: max memory allocated: {max_memory_allocated / GIGABYTE:.4f} GB")
    # print(f"{prefix}: max memory reserved: {max_memory_reserved / GIGABYTE:.4f} GB")
    
    with open(f'open_sora_serving.txt', 'a') as file:
        file.write(f"{num_frames}\t{resolution}\t{latency * 1000:.2f}\t{mem1:.2f}\t{mem2:.2f}\n")

ARGS = None

def main(args, text_encoder_policy=None, model_policy=None):
    
    FRAMES=args.num_frames
    RESOLUTION=args.resolution
    
    ARGS = args
    
    num_frames = int(args.num_frames)
    resolution = args.resolution
    
    prompts = [
        "A beautiful sunset over the city",
        # "The majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. The camera angle provides a bird's eye view of the waterfall, allowing viewers to appreciate the full height and grandeur of the waterfall. The video is a stunning representation of nature's power and beauty.",
        # "A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures.",
        # "A serene night scene in a forested area. The first frame shows a tranquil lake reflecting the star-filled sky above. The second frame reveals a beautiful sunset, casting a warm glow over the landscape. The third frame showcases the night sky, filled with stars and a vibrant Milky Way galaxy. The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. The style of the video is naturalistic, emphasizing the beauty of the night sky and the peacefulness of the forest.",
        # "A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. The scene is a blur of motion, with cars speeding by and pedestrians navigating the crosswalks. The cityscape is a mix of towering buildings and illuminated signs, creating a vibrant and dynamic atmosphere. The perspective of the video is from a high angle, providing a bird's eye view of the street and its surroundings. The overall style of the video is dynamic and energetic, capturing the essence of urban life at night.",
    ]
    device = get_current_device()
    free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
    print(f"Free memory on GPU: {free_memory / GIGABYTE:.4f} GB")
    print(FRAMES, RESOLUTION)

    module_config = _PRESET_MODULE_CONFIG_REGISTRY["v1.2-default"]

    if model_policy is not None:
        module_config.model_kwargs["enable_layernorm_kernel"] = False

    engine = Engine(
        module_config,
        LoadConfig(
            from_pretrained_vae=None,
            from_pretrained_text_encoder="DeepFloyd/t5-v1_1-xxl",
            from_pretrained_model=None,
            dtype="bf16",
            local_files_only=False,
        ),
        GenerationConfig(
            batch_size=1,
            condition_frame_length=5,
            align=5,
            multi_resolution="STDiT2",
            fps=24,
        ),
        text_encoder_policy=text_encoder_policy,
        model_policy=model_policy,
    )
    # print_memory_usage("After engine init", device)

    start_time = time.time()
    engine.generate(
        prompts=prompts,
        num_frames=num_frames,
        resolution=resolution,
        image_size=(480, 853),
        loop=1,
        reference_path=None,
        mask_strategy=None,
        save_dir="./samples/samples/",
    )
    record_time = time.time() - start_time
    print_memory_usage("After generation", device, record_time, num_frames, resolution)

FRAMES = ""
RESOLUTION = ""

if __name__ == "__main__":
    from opensora_serving.acceleration.shardformer.policies.stdit3 import STDIT3Policy
    from opensora_serving.acceleration.shardformer.policies.t5_encoder import T5EncoderPolicy

    text_encoder_policy = T5EncoderPolicy()
    model_policy = STDIT3Policy()

    # When using shardformer (provide any policy), use torchrun to run the script
    # torchrun --standalone --nproc_per_node 1 scripts/inference.py
    # main(text_encoder_policy=text_encoder_policy)
    # main(model_policy=model_policy)
    # main(text_encoder_policy=text_encoder_policy, model_policy=model_policy)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames')
    parser.add_argument('--resolution')

    args = parser.parse_args()
    
    main(args=args)
