#!/bin/bash
# Read input arguments and assign them to variables
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--nodes) NNODES="$2"; shift ;;
        -i|--node-id) NODE_ID="$2"; shift ;;
        -m|--master-addr) MASTER_ADDRR="$2"; shift ;;
        -p|--master-port) MASTER_PORT="$2"; shift ;;
        -d|--model-dir) MODEL_DIR="$2"; shift ;;
        -t|--tokenizer) TOKENIZER_PATH="$2"; shift ;;
        -dev|--device) DEVICE="$2"; shift ;;
        -pf|--prompt-file) PROMPT_FILE="$2"; shift ;;
        -task|--task) TASK="$2"; shift ;;
        -b|--batch) BATCH="$2"; shift ;;
        -temp|--temperature) TEMPERATURE="$2"; shift ;;
        -l|--max_seq_len) MAX_SEQ_LEN="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


if [ -n "$NNODES" ]; then
    # Multiple worker operation
    torchrun --nproc_per_node=1 \
            --nnodes=$NNODES \
            --node_rank=$NODE_ID \
            --master_addr=$MASTER_ADDRR \
            --master_port=$MASTER_PORT \
            $TASK \
            --ckpt_dir $MODEL_DIR \
            --tokenizer_path $TOKENIZER_PATH \
            --temperature $TEMPERATURE \
            --max_seq_len $MAX_SEQ_LEN \
            --max_batch_size $BATCH \
            --device $DEVICE \
            --prompts_file $PROMPT_FILE

else
    # Single worker operation 
    torchrun --nproc_per_node=1 \
            $TASK \
            --ckpt_dir $MODEL_DIR \
            --tokenizer_path $TOKENIZER_PATH \
            --temperature $TEMPERATURE \
            --max_seq_len $MAX_SEQ_LEN \
            --max_batch_size $BATCH \
            --device $DEVICE \
            --prompts_file $PROMPT_FILE

fi