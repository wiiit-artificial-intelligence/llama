#!/bin/bash

# Help function for the script
show_help() {
    echo "This is a script to perform LLaMa tasks using torchrun."
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "-w, --nproc-per-node          Number of worker in one node. Default: 1"
    echo "-n, --nodes                   Number of nodes."
    echo "-i, --node-id                 Node ID in the cluster."
    echo "-m, --master-addr             Master address. Should be the IP address of node 0."
    echo "-p, --master-port             Master port."
    echo "-d, --model-dir               Path to model/checkpoint directory."
    echo "-t, --tokenizer               Path of model tokenizer."
    echo "-task, --task                 Task to execute (chat or text)."
    echo "-device, --device             Device (cpu or cuda)."
    echo "-prompt-file, --prompt-file   File with prompts. Chechk examples in prompts/ folder."
    echo "-temperature, --temperature   Temperature of the model. Default 0.0 (deterministic inference)."
    echo "-b, --batch                   Batch size. Defaults value. Text: 4. Chat: 6."
    echo "-l, --max-seq-len             Maximum sequence length. Defaults value. Text: 128. Chat: 512."
    echo "-h, --help                    Display this help and exit."
    echo ""
    echo "Note:"
    echo "--nodes, --node-id, --master-addr, --master-port arguments should only be declared if you perform inference in more than one node."
    echo ""
    echo "Examples:"
    echo "Chat example in a single CPU worker:"
    echo "./run.sh -task chat -d /path/to/model -t /path/to/tokenizer -device cpu -prompt-file prompts/chat_completion_example.yml"
    echo ""
    echo "Chat example in two CPU workers:"
    echo "./run.sh -task chat -n 2 -i 0 -m <node-ip-address> -p 12345 -d /path/to/model -t /path/to/tokenizer -device cpu -prompt-file prompts/chat_completion_example.yml"
    echo ""
}

# If no arguments provided or help is requested, display the help
if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Read input arguments and assign them to variables
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -w|--nproc-per-node) NPROC_PER_NODE="$2"; shift ;;
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
        -l|--max-seq-len) MAX_SEQ_LEN="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


# Default values for temperature, batch, and max_seq_len based on task
case "$TASK" in
    "chat")
        EXEC_FILE="example_chat_completion.py"
        : ${NPROC_PER_NODE:=1}
        : ${BATCH:=6}
        : ${TEMPERATURE:=0.0}
        : ${MAX_SEQ_LEN:=512}
        ;;
    "text")
        EXEC_FILE="example_text_completion.py"
        : ${NPROC_PER_NODE:=1}
        : ${BATCH:=4}
        : ${TEMPERATURE:=0.0}
        : ${MAX_SEQ_LEN:=128}
        ;;
    *)
        echo "Unknown task: $TASK"
        exit 1
        ;;
esac


if [ -n "$NNODES" ]; then
    # Multiple node operation
    torchrun --nproc_per_node=$NPROC_PER_NODE \
            --nnodes=$NNODES \
            --node_rank=$NODE_ID \
            --master_addr=$MASTER_ADDRR \
            --master_port=$MASTER_PORT \
            $EXEC_FILE \
            --ckpt_dir $MODEL_DIR \
            --tokenizer_path $TOKENIZER_PATH \
            --temperature $TEMPERATURE \
            --max_seq_len $MAX_SEQ_LEN \
            --max_batch_size $BATCH \
            --device $DEVICE \
            --prompts_file $PROMPT_FILE

else
    # Single node operation 
    torchrun --nproc_per_node=$NPROC_PER_NODE \
            $EXEC_FILE \
            --ckpt_dir $MODEL_DIR \
            --tokenizer_path $TOKENIZER_PATH \
            --temperature $TEMPERATURE \
            --max_seq_len $MAX_SEQ_LEN \
            --max_batch_size $BATCH \
            --device $DEVICE \
            --prompts_file $PROMPT_FILE

fi