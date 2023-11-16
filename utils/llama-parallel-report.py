import os
import argparse
import sys
import torch
import time
import numpy as np
import subprocess
import re
from pprint import pprint

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

import llamablocks

local_rank = int(os.environ.get('LOCAL_RANK', 0))
if not 'LOCAL_RANK' in os.environ:
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

torch.distributed.init_process_group("gloo")

if local_rank > 0:
    sys.stdout = open(os.devnull, "w")

model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
print(f"Local rank: {local_rank+1}/ world size: {model_parallel_size}")

if not model_parallel_is_initialized():
    print("Init model parallel")
    initialize_model_parallel(model_parallel_size)

torch.device(local_rank)

def format_model_name(name):
    if not isinstance(name, str):
        raise ValueError("Model name must be a string!")
    
    name = name.lower()

    name_mapper = {
        '7b': 'llama2-7b',
        '13b': 'llama2-13b',
        '34b': 'llama2-34b',
        '70b': 'llama2-70b'
    }

    if name in name_mapper.keys():
        name = name_mapper[name]

    supported_models = ['llama2-7b', 'llama2-13b', 'llama2-34b', 'llama2-70b']
    if not name in supported_models:
        raise ValueError(f"Model {name} not supported. The supported models are {supported_models}.")
    
    return name

def estimate_computing_capability(dtype=None):
    default_dtype = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(torch.float32)

    M = 2**12
    N = 2**13
    P = 2**13
    n_repeat = 5
    with torch.no_grad():
        tensor1 = torch.randn(M, N)
        tensor2 = torch.randn(N, P)

        read_time_values = []
        exec_time_values = []
        write_time_values = []

        for _ in range(n_repeat):
            torch.distributed.barrier()
            start_time = time.time()
            tensor1_ = tensor1.clone()
            tensor2_ = tensor2.clone()
            tensor1_ = tensor1.clone()
            tensor2_ = tensor2.clone()
            tensor1_ = tensor1.clone()
            tensor2_ = tensor2.clone()
            torch.distributed.barrier()
            stop_time = time.time()
            read_time_values.append((stop_time - start_time)/3)

            torch.distributed.barrier()
            start_time = time.time()
            res = torch.matmul(tensor1_, tensor2_)
            res = torch.matmul(tensor1_, tensor2_)
            res = torch.matmul(tensor1_, tensor2_)
            torch.distributed.barrier()
            stop_time = time.time()
            exec_time_values.append((stop_time - start_time)/3)

            torch.distributed.barrier()
            start_time = time.time()
            res_ = res.clone()
            res_ = res.clone()
            res_ = res.clone()
            torch.distributed.barrier()
            stop_time = time.time()
            write_time_values.append((stop_time - start_time)/3)

    read_time_mean = np.mean(read_time_values)
    exec_time_mean = np.mean(exec_time_values)
    write_time_mean = np.mean(write_time_values)
    memory_acces_time = np.max([read_time_mean, write_time_mean])

    if memory_acces_time >= exec_time_mean:
        print("Warning: the memory access time is bigger than computing time!")
    elif memory_acces_time >= 0.5 * exec_time_mean:
        print("Warning: the memory access time is near of computing time!")

    theorical_ops = 2 * M * N * P
    computing_capability = theorical_ops/exec_time_mean
    computing_capability_std = np.std(theorical_ops/np.array(exec_time_values))
    computing_capability_max = np.max(theorical_ops/np.array(exec_time_values))

    if computing_capability_std > 0.1 * computing_capability:
        print("Warning: the computing capability estimation has high variance!")

    torch.set_default_dtype(default_dtype)
    return computing_capability, computing_capability_std, computing_capability_max

def parse_dmidecode_output(output):
    # Define a regular expression pattern to extract key-value pairs
    pattern = re.compile(r'^\s+([^:]+):\s*(.*)$')

    # Initialize an empty list to store dictionaries for each RAM device
    ram_devices = []
    current_device = {}

    # Iterate over each line in the output
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            # Extract key-value pairs and add them to the current device dictionary
            key, value = match.groups()
            current_device[key.strip()] = value.strip()
        elif not line.strip() and current_device:
            # Empty line indicates the end of one RAM device
            ram_devices.append(current_device)
            current_device = {}

    # Add the last device if any
    if current_device:
        ram_devices.append(current_device)

    return ram_devices

def execute_dmidecode_command():
    # Execute the dmidecode command using subprocess
    command = "sudo dmidecode --type 17"
    command_nosudo = "dmidecode --type 17"
    try:
        # Run the command and capture the output
        output = subprocess.check_output(command, shell=True, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        output = None

    if output is None:
        try: 
            output = subprocess.check_output(command_nosudo, shell=True, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            return None

    # Parse the output and convert it to a list of dictionaries
    result_list = parse_dmidecode_output(output)

    return result_list

def read_memory_bandwidth(memory_info):
    memory_channels = len(memory_info)

    # we assume that all memory devices are the same
    if "Configured Memory Speed" in memory_info[0].keys():
        memory_speed_info = memory_info[0]["Configured Memory Speed"].split()
    elif "Speed" in memory_info[0].keys():
        memory_speed_info = memory_info[0]["Speed"].split()
    else:
        raise ValueError("Memory speed not available!")
    
    memory_speed = float(memory_speed_info[0])
    if memory_speed_info[1] == "kT/s":
        memory_speed *= 1e3
    elif memory_speed_info[1] == "MT/s":
        memory_speed *= 1e6
    elif memory_speed_info[1] == "GT/s":
        memory_speed *= 1e9
    else:
        raise ValueError("Unknown memory speed unit")
    
    if "Data Width" in memory_info[0].keys():
        memory_data_info = memory_info[0]["Data Width"].split()
    else:
        raise ValueError("Memory data width not available!")
    
    memory_data_width = int(memory_data_info[0])
    if memory_data_info[1] == "bits":
        pass
    elif memory_data_info[1] == "bytes":
        memory_data_width *= 8
    else:
        raise ValueError("Unknown memory data width unit")
    
    memory_bw = memory_channels * memory_speed * memory_data_width
    return memory_bw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="llama2-70b",
        help="Name of the model. Example: llama2-70b or 70b.")
    parser.add_argument(
        "-f", "--flops", type=float, default=0.0,
        help="Computing capability of a single worker. The value must be in GFLOPS.")
    parser.add_argument(
        "-d", "--memory_bw", type=float, default=0.0,
        help="Memory access bandwidth of a single worker. The value must be in Gbps.")
    parser.add_argument(
        "-l", "--network_latency", type=float, default=0.0,
        help="Network latency (packet traveling time between workers). Value must be in microseconds [us].")
    parser.add_argument(
        "-n", "--network_bw", type=float, default=0.0,
        help="Network bandwidth (packets transfer speed between workers). Value must be in Gbps.")
    parser.add_argument(
        "-p", "--n_workers", type=int, default=None,
        help="Number of workers for calculation in the cluster.")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1,
        help="Batch size (number of simultaneous input prompts).")
    parser.add_argument(
        "-i", "--seq_ini_len", type=int, default=128,
        help="Number of tokens of the maximum input sequence.")
    parser.add_argument(
        "-o", "--output_tokens", type=int, default=512,
        help="Maximum number of tokens to generate.")
    parser.add_argument(
        "-t", "--threads", type=int, default=None,
        help="Maximum number of threads to use at each worker. Default: all.")
    
    args = parser.parse_args()

    # threads
    num_threads = args.threads
    if num_threads is not None:
        torch.set_num_threads(num_threads)
        print(f"{'Num of threads':32.32s} = {num_threads}")

    # data type
    dtype = torch.float32
    torch.set_default_dtype(torch.float32)
    print(f"{'Data type':32.32s}: {str(dtype)} - Item size = {dtype.itemsize} Bytes")

    # set model
    model_name = format_model_name(args.model)
    print(f"{'Model':32.32s} = {model_name}")
    model_args = llamablocks.get_model_args(model_name)
    pprint(model_args.__dict__)

    # computing capability
    worker_flops = args.flops*1e9
    if worker_flops == 0.0:
        print(" > Estimating computing capability")
        worker_flops, _, _ = estimate_computing_capability(dtype)
    print(f"{'Computing capability':32.32s} = {worker_flops/1e9:.3f} [GFLOPS]")

    # memory access bandwidth
    memory_bw = args.memory_bw*1e9
    if memory_bw == 0.0:
        # Get the memory information.
        memory_info = execute_dmidecode_command()

        if memory_info is not None:
            memory_bw = read_memory_bandwidth(memory_info)

    if memory_bw == 0.0:
        print(f"{'Memory bandwidth':32.32s} = No value available => Infinite bandwidth") 
        print('WARNING! The optimal number of processors cannot be estimated without memory bandwidth information.')
        exit()
    else:
        print(f"{'Memory bandwidth':32.32s} = {memory_bw/1e9:.3f} [Gbps]")

    # network latency
    network_latency = args.network_latency*1e-6
    if network_latency == 0.0:
        print(f"{'Network latency':32.32s} = Not specified => 0.0 [us]")
    else:
        print(f"{'Network latency':32.32s} = {network_latency*1e6:.3f} [us]")

    # network bandwidth
    network_bw = args.network_bw*1e9
    if network_bw == 0.0:
        print(f"{'Network bandwidth':32.32s} = Not specified => infinite bandwidth")
    else:
        print(f"{'Network bandwidth':32.32s} = {network_bw/1e9:.3f} [Gbps]")

    # sequences parameters
    batch_size = args.batch_size
    print(f"{'Batch size':32.32s} = {batch_size}")

    seq_ini_len = args.seq_ini_len
    print(f"{'Init sequence lenght':32.32s} = {seq_ini_len}")

    output_tokens = args.output_tokens
    print(f"{'Output tokens':32.32s} = {output_tokens}")

    # number of workers
    n_workers_cluster = args.n_workers
    if n_workers_cluster is None:
        n_workers_cluster = model_parallel_size
    if n_workers_cluster != model_parallel_size:
        print(f"Warning: the number of cluster you provided ({n_workers_cluster}) is different from world size ({model_parallel_size})")
    print(f"{'Num of workers':32.32s} = {n_workers_cluster}")

    # Optimal number of workers
    loop_j = 1
    n_workers_optimal = llamablocks.optimal_num_workers(
            args=model_args,
            worker_flops=worker_flops,
            memory_bw=memory_bw,
            network_bw=network_bw,
            network_latency=network_latency,
            dtype=dtype,
            batch_size=batch_size,
            seq_ini_len=seq_ini_len,
            loop_j=loop_j
        )
    print(f"{'Optimal num of workers':32.32s} = {n_workers_optimal}")

    # layer theorical time
    theorical_layer_time_params = {
        "args": model_args,
        "n_workers": None,
        "worker_flops": worker_flops,
        "memory_bw": memory_bw,
        "network_bw": network_bw,
        "network_latency": network_latency,
        "dtype": dtype,
        "batch_size": batch_size,
        "seq_ini_len": seq_ini_len,
        "loop_j": None        
    }

    print(" --- Layer Time (token #1)---")
    
    ## single worker
    theorical_layer_time_params.update({"n_workers": 1, "loop_j": 0})
    layer_time_theorical_single, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
    print(f"{'Theorical - single worker':32.32s} = {layer_time_theorical_single*1e3:8.3f} [ms]")

    ## cluster workers
    theorical_layer_time_params.update({"n_workers": n_workers_cluster, "loop_j": 0})
    layer_time_theorical_cluster, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
    if n_workers_cluster > 1:
        print(f"{'Theorical - cluster':32.32s} = {layer_time_theorical_cluster*1e3:8.3f} [ms] ({n_workers_cluster} workers)")

    ## optimal workers
    theorical_layer_time_params.update({"n_workers": n_workers_optimal, "loop_j": 0})
    layer_time_theorical_optimal, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
    print(f"{'Theorical - optimal':32.32s} = {layer_time_theorical_optimal*1e3:8.3f} [ms] ({n_workers_optimal} workers)")

    print(" --- Layer Time (token #2)---")
    
    ## single worker
    theorical_layer_time_params.update({"n_workers": 1, "loop_j": 1})
    layer_time_theorical_single, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
    print(f"{'Theorical - single worker':32.32s} = {layer_time_theorical_single*1e3:8.3f} [ms]")

    ## cluster workers
    theorical_layer_time_params.update({"n_workers": n_workers_cluster, "loop_j": 1})
    layer_time_theorical_cluster, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
    if n_workers_cluster > 1:
        print(f"{'Theorical - cluster':32.32s} = {layer_time_theorical_cluster*1e3:8.3f} [ms] ({n_workers_cluster} workers)")

    ## optimal workers
    theorical_layer_time_params.update({"n_workers": n_workers_optimal, "loop_j": 1})
    layer_time_theorical_optimal, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
    print(f"{'Theorical - optimal':32.32s} = {layer_time_theorical_optimal*1e3:8.3f} [ms] ({n_workers_optimal} workers)")

    print(f" --- Layer Time (token #{output_tokens})---")
    
    ## single worker
    theorical_layer_time_params.update({"n_workers": 1, "loop_j": output_tokens-1})
    layer_time_theorical_single, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
    print(f"{'Theorical - single worker':32.32s} = {layer_time_theorical_single*1e3:8.3f} [ms]")

    ## cluster workers
    theorical_layer_time_params.update({"n_workers": n_workers_cluster, "loop_j": output_tokens-1})
    layer_time_theorical_cluster, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
    if n_workers_cluster > 1:
        print(f"{'Theorical - cluster':32.32s} = {layer_time_theorical_cluster*1e3:8.3f} [ms] ({n_workers_cluster} workers)")

    ## optimal workers
    theorical_layer_time_params.update({"n_workers": n_workers_optimal, "loop_j": output_tokens-1})
    layer_time_theorical_optimal, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
    print(f"{'Theorical - optimal':32.32s} = {layer_time_theorical_optimal*1e3:8.3f} [ms] ({n_workers_optimal} workers)") 

    # Model time

    print(f" --- Model Minimum Inference Time ({output_tokens} tokens)---")

    inference_time_values_single = []
    inference_time_values_cluster = []
    inference_time_values_optimal = []

    for j in range(output_tokens):
        ## single worker
        theorical_layer_time_params.update({"n_workers": 1, "loop_j": j})
        layer_time_theorical_single, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
        inference_time_values_single.append(model_args.n_heads * layer_time_theorical_single)

        ## cluster workers
        theorical_layer_time_params.update({"n_workers": n_workers_cluster, "loop_j": output_tokens-1})
        layer_time_theorical_cluster, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
        inference_time_values_cluster.append(model_args.n_heads * layer_time_theorical_cluster)

        ## optimal workers
        theorical_layer_time_params.update({"n_workers": n_workers_optimal, "loop_j": output_tokens-1})
        layer_time_theorical_optimal, _, _, _ = llamablocks.theorical_layer_time(**theorical_layer_time_params)
        inference_time_values_optimal.append(model_args.n_heads * layer_time_theorical_optimal)

    print(f"{'Total theorical - single worker':32.32s} = {np.sum(inference_time_values_single):8.3f} [s]")
    if n_workers_cluster > 1:
        print(f"{'Total theorical - cluster':32.32s} = {np.sum(inference_time_values_cluster):8.3f} [s] ({n_workers_cluster} workers)")
    print(f"{'Total theorical - optimal':32.32s} = {np.sum(inference_time_values_optimal):8.3f} [s] ({n_workers_optimal} workers)")

    print(f"{'Throughput - single worker':32.32s} = {output_tokens/np.sum(inference_time_values_single[1:]):6.1f}   [tokens/s]")
    if n_workers_cluster > 1:
        print(f"{'Throughput - cluster':32.32s} = {output_tokens/np.sum(inference_time_values_cluster[1:]):6.1f}   [tokens/s]")
    print(f"{'Throughput - optimal':32.32s} = {output_tokens/np.sum(inference_time_values_optimal[1:]):6.1f}   [tokens/s]")