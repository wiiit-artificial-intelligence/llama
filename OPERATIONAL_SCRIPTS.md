# Infrastructure Setup and Operational Scripts Overview

⚠️ **These scripts have only been tested in Azure VMs, so their use in other cloud providers is not guaranteed.**

## Overview
After setting up the infrastructure, three scripts streamline **worker configuration**, **application deployment**, and **distributed inference** within the distributed environment.

- [`configure_worker.sh`](run/configure_workers.sh)
    - Configures each node/worker by installing necessary dependencies for running LLaMA model inferences.
        - Installs Docker, Docker-compose, and, for GPU nodes, NVIDIA drivers and utilities.
    - Clones this repository and downloads the model checkpoint[^1].
    - Starts the corresponding container based on the declared platform (CPU/GPU).

- [`shard_model_checkpoint.sh`](run/shard_model_checkpoint.sh)
    - Facilitates sharding of the original model checkpoints into a configurable number, typically equivalent to the number of nodes/workers.
    - Currently, this sharding task is performed individually on all workers[^2].

- [`distributed_inference.sh`](run/distributed_inference.sh)
    - Initiates distributed inference using torchrun on each worker.

## Usage Instructions and Considerations
- The scripts execute commands remotely on the workers from your local machine.
- Before running the scripts on your machine, create a `.env` file based on the provided [`.env.sample`](run/.env.sample). Each variable's significance is commented.
- Follow the sequence of execution:
    - `bash configure_worker.sh`: for worker configuration.
    - `bash shard_model_checkpoint.sh`: to shard model checkpoints based on a specific configuration.
    - `bash distributed_inference.sh`: to initiate distributed inference.

## Considerations for Execution
`configure_worker.sh` ideally runs once, while `shard_model_checkpoint.sh` executes only when intending to perform inferences on a different number of workers than the original model's checkpoints.


[^1]: Currently, this process is duplicated across all workers. Future enhancement includes downloading the model checkpoint on a single worker and uploading it to shared storage.

[^2]: A recommended improvement involves performing this task on one worker and uploading the shards to shared storage, similar to the model checkpoints.
