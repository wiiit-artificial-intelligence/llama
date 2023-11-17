# Llama 2

We are unlocking the power of large language models. Our latest version of Llama is now accessible to individuals, creators, researchers and businesses of all sizes so that they can experiment, innovate and scale their ideas responsibly. 

This release includes model weights and starting code for pretrained and fine-tuned Llama language models — ranging from 7B to 70B parameters.

This repository is intended as a minimal example to load [Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) models and run inference. For more detailed examples leveraging Hugging Face, see [llama-recipes](https://github.com/facebookresearch/llama-recipes/).

## Updates post-launch

See [UPDATES.md](UPDATES.md). Also for a running list of frequently asked questions, see [here](https://github.com/facebookresearch/llama/blob/main/FAQ.md).

## Download

⚠️ **7/18: We're aware of people encountering a number of download issues today. Anyone still encountering issues should remove all local files, re-clone the repository, and [request a new download link](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). It's critical to do all of these in case you have local corrupt files.**

In order to download the model weights and tokenizer, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then to run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as `403: Forbidden`, you can always re-request a link.

### Access on Hugging Face

We are also providing downloads on [Hugging Face](https://huggingface.co/meta-llama). You must first request a download from the Meta website using the same email address as your Hugging Face account. After doing so, you can request access to any of the models on Hugging Face and within 1-2 days your account will be granted access to all versions.

## Quick Start

You can follow the steps below to quickly get up and running with Llama 2 models. These steps will let you run inference (locally or not).

1. Clone this repository.

2. If you are in a CPU, raise the CPU container. Otherwise, raise the GPU container. In any case, you need to have docker and docker-compose installed.
    ```bash
    cd .devcontainer

    docker-compose -f docker-compose_cpu.yml up -d
    ```
3. Open an interactive terminal in the container. 

    ```bash
    docker exec -it devcontainer-llama-dev-1 /bin/bash
    ```
    At this point, the worker it's ready to make inference. Now, you need to download the model.

4. Visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and register to download the model/s.

5. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

6. Once you get the email, navigate to your downloaded llama repository and run the download.sh script. 
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email. 
    - Do not use the “Copy Link” option but rather make sure to manually copy the link from the email.

7. You can test the environment running 7B model (inside container) using the command below:

```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
         --ckpt_dir <checkpoint_directory> \
         --tokenizer_path <tokenizer_path> \
         --max_seq_len 2048 \ 
         --max_batch_size 1 \
         --device cpu 
```

**Note**
- Replace  `<checkpoint_directory>` with the path to your checkpoint directory and `<tokenizer_path>` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.
- This example runs the [example_chat_completion.py](example_chat_completion.py) found in this repository but you can change that to a different .py file.
- In case of distributed inference in more than one worker, you must repeat the above steps in each of the workers in your cluster.

## Inference

### Distributed inference with `nnode` > 1
1. If you wanna run LLaMa model in a generic infraestructure with more than one node, you need to shard the model checkpoints (you must have downloaded the model previously). The number of shards, would be the number of worker in the cluster. <br>
As an example, if you have a cluster with 2 nodes, to shard LLaMa 7B checkpoint you should run (in each worker):
```bash
python utils/shard_model_checkpoint.py -n 2 -i llama-2-7b-chat/ -o llama-2-7b-chat-2-workers/
```

2. Once you have the shards in each worker you should run:

- In worker-0
```bash
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<ip-address-node-0> --master_port=12345 example_chat_completion.py --ckpt_dir llama-2-7b-chat-2-workers/  --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 1 --device cpu
```

- In worker-1
```bash
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<ip-address-node-0> --master_port=12345 example_chat_completion.py --ckpt_dir llama-2-7b-chat-2-workers/  --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 1 --device cpu
```

### Metrics

Below are useful metrics to measure inference speed. Assuming $T$ is the total time, $B$ is the batch size, $L$ is the decoded sequence length.

#### Latency Definition
Latency is the time it takes to get the decoded result at target length $L$, regardless of the batch size $B$. Latency represents how long the user should wait to get the response from the generation model.

$$ \text{Latency [s]} = T $$

#### Per-token latency
One step of autoregressive decoding generates a token for each sample in the batch. Per-token latency is the average time for that one step.

$$ \text{Per-token latency [s/token]}= \frac{T}{L} $$

#### Throughput
Throughput measures how many tokens are generated per unit time. While it’s not a useful metric for evaluating online serving it is useful to measure the speed of batch processing.

$$ \text{Throughput [tokens/s]} = \frac{B * L}{T} $$


After running distributed inference, you'll see something like this as inference metrics:

```bash
------ Inference metrics ------
Generated tokens: 64
Latency: 21.22 (s).
Per-token latency: 0.33 (s/token)
Throughput: 3.02 (tokens/s)
-------------------------------
```
