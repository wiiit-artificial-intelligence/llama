#!/bin/bash
# This script:
#   - install Docker, Docker-compose.
#   - clone distributed_inference branch from wiiit-llama repository and
#   - raise docker container to allow distributed inference in CPU cluster.
# The scripts only configure the worker. To launch inference, cjeck: launch_distributed_inferece.sh
#
# References:
#  - Docker: https://docs.privacera.com/latest/platform/pm-ig/install_docker_and_docker_compose_azure_ubuntu/ 

# Load variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "The .env file not exist"
    exit 1
fi



for REMOTE_HOST in "${SERVERS[@]}"
do
    echo "Accessing to $REMOTE_HOST..."
    sshpass -P "passphrase" -p "$PASSWORD" ssh "$REMOTE_HOST" "bash -s" <<ENDSSH
    
    ### VM identification ###
    echo "Virtual machine \$(hostname) user: \$(whoami)"
    
    ### Install Docker ###
    echo "Installing docker..."  
    sudo apt update  
    sudo apt install docker.io -y
    sudo service docker start
    sudo usermod -a -G docker \$(whoami)
    echo "Docker succesfully installed!"

    if [ "$DEVICE" = "cuda" ]; then
        sudo apt-get update
        sudo apt install -y linux-modules-nvidia-535-azure nvidia-driver-535
    fi

    sudo reboot
    
ENDSSH
    
    sleep 45   
    sshpass -P "passphrase" -p "$PASSWORD" ssh "$REMOTE_HOST" 'bash -s' <<ENDSSH
    
    ### Confirm docker installation ###
    docker info

    if [ "$DEVICE" = "cuda" ]; then
        nvidia-smi
    fi

 
    ### Install docker-compose installation ###
    echo "Installing docker-compose..."  
    sudo  curl -L $DOCKER_COMPOSE_URL
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker-compose succesfully installed!"

    # If worker has cuda, install NVIDIA container toolkit 
    if [ "$DEVICE" = "cuda" ]; then

        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
        && sudo apt-get update

        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker

    fi
	
    ### Creates workspace folder to clone llama repository ###
    echo "Cloning repository..."	
    mkdir workspace
    cd workspace
   
    git clone --branch "$REPO_BRANCH" "$REPO_URL"
    cd llama

    ### Download LLaMa model
    export PRESIGNED_URL="$META_URL"
    export MODEL_SIZE="$MODEL_VERSION"
    bash download.sh
    
    ### Start docker container ###
    cd docker

    # Raise container
    if [ "$DEVICE" = "cuda" ]; then
        docker-compose -f docker-compose_gpu.yml up -d
    else
        docker-compose -f docker-compose_cpu.yml up -d
    fi
    docker ps
    
ENDSSH
done
