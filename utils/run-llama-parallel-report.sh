## Full parametrization
# python llama-parallel-report.py \
#     --model llama2-70b \
#     --flops 612.0 \
#     --memory_bw 312.0 \
#     --network_latency 38.0 \
#     --network_bw 12.5 \
#     --n_workers 4 \
#     --batch_size 1 \
#     --seq_ini_len 128 \
#     --output_tokens 512

## Get CPU FLOPS and Memory Access Bandwidth
python llama-parallel-report.py \
    --model llama2-7b \
    --network_latency 150.0 \
    --network_bw 12.5 \
    --n_workers 4 \
    --batch_size 1 \
    --seq_ini_len 2048 \
    --output_tokens 512

## Get CPU FLOPS and Memory Access Bandwidth (with threads limited)
# python llama-parallel-report.py \
#     --model llama2-70b \
#     --network_latency 38.0 \
#     --network_bw 12.5 \
#     --n_workers 4 \
#     --batch_size 1 \
#     --seq_ini_len 128 \
#     --output_tokens 512 \
#     --threads 2