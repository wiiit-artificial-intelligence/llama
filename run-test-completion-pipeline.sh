torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=10.100.192.15 \
    --master_port=12345 \
    example_text_completion.py \
    --ckpt_dir models/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 \
    --max_batch_size 4 \
    --prompts_file prompts/text_completion_example.yml \
    --temperature 0 \
    --load_weights False \
    --model_flavor pipellama2
