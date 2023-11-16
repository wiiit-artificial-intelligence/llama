import os
import sys
import torch
import time
import numpy as np

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

local_rank = int(os.environ['LOCAL_RANK'])
torch.distributed.init_process_group("gloo")

if local_rank > 0:
    sys.stdout = open(os.devnull, "w")

model_parallel_size = int(os.environ.get("WORLD_SIZE"))
print(f"{local_rank+1}/{model_parallel_size}")

if not model_parallel_is_initialized():
    print("Init model parallel")
    initialize_model_parallel(model_parallel_size)

torch.device(local_rank)
torch.set_default_dtype(torch.float32)

n_repeat = 20
M = 2**13
N = 2**13
P = 2**13

with torch.no_grad():
    tensor1 = torch.randn(M, N)
    tensor2 = torch.randn(N, P)
    print(f"Memory usage = {(M*N + N*P)*4/1e6:.3f} [MBytes]")
    time.sleep(2.0)

    n_repeat_ = n_repeat - int(np.log2(max(M, N, P))) - 1
    loading_time_values = []

    for _ in range(100):
        start_time = time.time()
        tensor1_ = tensor1.clone().detach()
        tensor2_ = tensor2.clone().detach()
        loading_time = (time.time() - start_time)
        loading_time_values.append(loading_time)

    loading_time_per_element = np.mean(loading_time_values)/(2*(M*N + N*P))

    print(f"Loading mean time (total)     = {np.mean(loading_time_values)/2:.3f} [s]")
    print(f"Loading mean time per element = {loading_time_per_element*1e9:.3f} [ns/element]")
    print(f"Loading mean time per byte    = {loading_time_per_element*1e9/4:.3f} [ns/Byte] ({4*8*1e-9/loading_time_per_element:.3f} [Gbps])")

    time.sleep(2.0)
    res = torch.zeros((M, P))

    n_repeat_ = n_repeat - int(np.log2(max(M, N, P))) - 1
    start_time = time.time()
    for _ in range(n_repeat_):
        res = torch.matmul(tensor1_, tensor2_)
    mult_time = (time.time() - start_time)/n_repeat_

    theorical_ops = 2 * M * N * P

    print(f"Matrix multiplication result size         = {res.size()}")
    print(f"Matrix multiplication teorical ops        = {theorical_ops/1e9:.3f} [GFLOPS]")
    print(f"Matrix multiplication execution mean time = {mult_time:.3f} [s]")
    print(f"Computing capability = {theorical_ops*1e-9/mult_time:.3f} [GFLOPS/s]")
