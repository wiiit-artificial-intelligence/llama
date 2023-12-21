#!/bin/bash
# This script  launch inferece over cluster of CPU.

# Load variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "The .env file not exist"
    exit 1
fi


if [ "$NCORES" != "All" ]; then
    DOCKER_COMMAND_BASE="export OMP_NUM_THREADS=$NCORES && bash run.sh \
        --task $TASK \
        --nproc-per-node 1 \
        --model-dir $SHARD_PATH \
        --tokenizer $TOKENIZER_PATH \
        --device $DEVICE \
        --prompt-file $PROMPT_FILE \
        --batch $BATCH \
        --do-profile $DO_PROFILE \
        --profile-output $PROFILE_OUTPUT"
else
    DOCKER_COMMAND_BASE="bash run.sh \
        --task $TASK \
        --nproc-per-node 1 \
        --model-dir $SHARD_PATH \
        --tokenizer $TOKENIZER_PATH \
        --device $DEVICE \
        --prompt-file $PROMPT_FILE \
        --batch $BATCH \
        --do-profile $DO_PROFILE \
        --profile-output $PROFILE_OUTPUT"
fi
	   
# Total number of nodes
NODES=${#SERVERS[@]}
# NodeId
NODE_ID=$NODES

for REMOTE_HOST in "${SERVERS[@]}"
do
    ((NODE_ID--)) 
    echo "Nodes: $NODES - NodeID: $NODE_ID - NodeName: $REMOTE_HOST"
         
    # Update base command using distributed information   
    if [ $DEVICE = 'cpu' ]; then 	
        DOCKER_COMMAND="$DOCKER_COMMAND_BASE \
                    --nodes $NODES \
                    --node-id $NODE_ID \
                    --master-addr $MASTER_ADDR \
                    --master-port $MASTER_PORT"
    else
        DOCKER_COMMAND=$DOCKER_COMMAND_BASE
    fi
                   
    # All the nodes runs in background except the NodeId=0
    if [ $NODE_ID -gt 0 ]; then
        sshpass -P "passphrase" -p "$PASSWORD" ssh "$REMOTE_HOST" "bash -s" <<ENDSSH
        
        cd "$REMOTE_DIR"
        docker exec -d "$DOCKER_CONTAINER_NAME" /bin/bash -c "$DOCKER_COMMAND"
ENDSSH
    else
        sshpass -P "passphrase" -p "$PASSWORD" ssh "$REMOTE_HOST" "bash -s" <<ENDSSH

        cd "$REMOTE_DIR"             
        docker exec -t "$DOCKER_CONTAINER_NAME" /bin/bash -c "$DOCKER_COMMAND"
ENDSSH
    fi
done