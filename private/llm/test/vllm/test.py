from vllm import LLM, SamplingParams
import argparse
import torch
import time

GIGABYTE = 1024**3

def get_current_device():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    print(f"Using device: {device}")

def prepare_data(batch_size, seq_len):
    input_ids = torch.randint(10, 30000, (batch_size, seq_len), device=get_current_device())
    return input_ids

def print_details_info(whole_end2end, total_token_num, input_len, output_len, batch_size):
    whole_avg_latency = whole_end2end / (total_token_num)

    end2end_latency = whole_end2end * 1000
    per_token_latency = whole_avg_latency * 1000
    token_num = total_token_num
    throughput = token_num / whole_end2end
    mem1 = torch.cuda.max_memory_allocated(device=get_current_device()) / GIGABYTE
    mem2 = torch.cuda.memory_reserved(device=get_current_device()) / GIGABYTE

    with open('vllm_results.txt', 'a') as file:
        file.write(f"{end2end_latency:.2f}\t{per_token_latency:.2f}\t{token_num}\t{throughput:.2f}\t{mem1:.2f}\t{mem2:.2f}\t{input_len}\t{output_len}\t{batch_size} \n")
    

def main(args: argparse.Namespace): 
    model = args.model
    tp_size = args.tensor_parallel_size
    input_len = args.input_len
    output_len = args.output_len
    input_ids = prepare_data(args.batch_size, input_len).tolist()

    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=input_len + output_len,
        enforce_eager=True,
        device="cuda",
        dtype="float16",
    )
    
    num_requests = len(input_ids)
    
    for i in range(num_requests):
        sampling_params = SamplingParams(
            top_p=1.0,
            ignore_eos=True,
            max_tokens=output_len,
        )
        llm._add_request(
            prompt=None,
            prompt_token_ids=input_ids[i],
            sampling_params=sampling_params,
        )

    torch.cuda.synchronize()
    start_time = time.time()
    output = llm._run_engine(use_tqdm=True)
    torch.cuda.synchronize()
    latency = time.time() - start_time
    total_num_tokens = sum([len(out.outputs[0].token_ids) for out in output])
    print_details_info(latency, total_num_tokens, input_len, output_len, args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--batch-size",
                        type=int,
                        default=None)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    main(args)