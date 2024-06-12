import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

import time, argparse

import os

GIGABYTE = 1024**3

def get_current_device():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device

def print_details_info(whole_end2end, total_token_num, input_len, output_len, batch_size):

    whole_avg_latency = whole_end2end / (total_token_num)

    end2end_latency = whole_end2end * 1000
    per_token_latency = whole_avg_latency * 1000
    token_num = total_token_num
    throughput = token_num / whole_end2end
    mem1 = torch.cuda.max_memory_allocated(device=get_current_device()) / GIGABYTE
    mem2 = torch.cuda.memory_reserved(device=get_current_device()) / GIGABYTE

    with open(f'{MODEL_NAME}_transformers.txt', 'a') as file:
        file.write(f"{input_len}\t{output_len}\t{batch_size}\t{end2end_latency:.2f}\t{per_token_latency:.2f}\t{token_num}\t{throughput:.2f}\t{mem1:.2f}\t{mem2:.2f}\n")
    
def data_generate(input_len, batch_size):
    input_ids = torch.randint(10, 30000, (batch_size, input_len), device=get_current_device())
    return input_ids

def generate(input_len, output_len, batch_size):    
    if NUM_GPUS == 1:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            ).to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATH,
            trust_remote_code=True,
            device='auto',
            torch_dtype=torch.float16,
            )
    
    input_ids = data_generate(input_len, batch_size).to('cuda')

    start_time = time.time()
    outputs = model.generate(
        inputs=input_ids,
        max_new_tokens=output_len,
    )
    record_time = time.time() - start_time
    total_num_tokens = outputs.size()[0] * outputs.size()[1] - input_len * batch_size
    
    print_details_info(record_time, total_num_tokens, input_len, output_len, batch_size)


def arg_parse():
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="test demo.")
    parser.add_argument("--input_len")
    parser.add_argument("--output_len")
    parser.add_argument("--batch_size")
    parser.add_argument("--model_path")
    parser.add_argument("--model_name")
    parser.add_argument("--num_gpus")
    return parser.parse_args()

MODEL_PATH = ""
MODEL_NAME = ""
NUM_GPUS = -1

if __name__ == "__main__":
    
    args = arg_parse()
    
    input_len = int(args.input_len)
    output_len = int(args.output_len)
    batch_size = int(args.batch_size)
    MODEL_PATH = args.model_path
    MODEL_NAME = args.model_name
    NUM_GPUS = int(args.num_gpus)
    
    print(f"Input len: {input_len}, Output len: {output_len}, Batch size: {batch_size}")
    generate(
        input_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
        )
