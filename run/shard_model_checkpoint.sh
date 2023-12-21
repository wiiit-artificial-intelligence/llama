#!/bin/bash
# This script shard model checkpoint for distributed inferece over cluster of CPU.

# Load variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "The .env file not exist"
    exit 1
fi
  
# Total number of nodes
NODES=${#SERVERS[@]}
# NodeId
NODE_ID=$NODES

for REMOTE_HOST in "${SERVERS[@]}"
do
    ((NODE_ID--)) 

    echo "Nodes: $NODES - NodeID: $NODE_ID - NodeName: $REMOTE_HOST"         
                  
    sshpass -P "passphrase" -p "$PASSWORD" ssh "$REMOTE_HOST" 'bash -s' <<ENDSSH
    
    cd "$REMOTE_DIR"

    if [ -d "$SHARD_PATH" ]; then
        echo "Shards for $NWORKERS already exist"
    else
        echo "Sharding model checkpoint in $NWORKERS shards"
        # Shard model checkpoint
        docker exec -t "$DOCKER_CONTAINER_NAME" /bin/bash -c "python utils/shard_model_checkpoint.py -n $NWORKERS -i $SOURCE_CHECKPOINT -o $SHARD_PATH"
    fi
         
ENDSSH
done