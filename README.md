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

You can follow the steps below to quickly get up and running with Llama 2 models. These steps will let you run inference. In case of distributed inference in more than one worker, you must repeat the following steps in each of the workers in your cluster.

1. Clone and download this repository.

2. If you are in a CPU cluster, raise the CPU container. Otherwise, raise the GPU container. In any case, you need to have docker and docker-compose installed on each worker.
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

7. Once the model/s you want have been downloaded, you can run the model in the container using the command below:
```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir <checkpoint_directory> \
    --tokenizer_path <tokenizer_path> \
    --max_seq_len 512 --max_batch_size 1 --device cpu 
```

**Note**
- Replace  `<checkpoint_directory>` with the path to your checkpoint directory and `<tokenizer_path>` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.
- This example runs the [example_chat_completion.py](example_chat_completion.py) found in this repository but you can change that to a different .py file.

## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 70B    | 8  |

All models support sequence length up to 4096 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware. <br>
This model-parallel represent the number of workers in one node.

### Distributed inference with `nnode` > 1
1. If you wanna run your LLaMa model in a generic infraestructure with more than one node, you need to shard the model checkpoints. The number of shards, would be the number of worker in the cluster. <br>
As an example, if you have a cluster with 2 workers, to shard LLaMa 7B checkpoint you should run (in each worker):
```bash
python utils/shard_model_checkpoint.py -n 2 -i llama-2-7b-chat/ -o llama-2-7b-chat-2-workers/
```

2. Once you have the shards in each worker you should run:

- In worker-0
```bash
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=10.1.133.12 --master_port=12345 example_chat_completion.py --ckpt_dir llama-2-7b-chat-2-workers/  --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 1 --device cpu
```

- In worker-1
```bash
torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=10.1.133.12 --master_port=12345 example_chat_completion.py --ckpt_dir llama-2-7b-chat-2-workers/  --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 1 --device cpu
```

### Pretrained Models

These models are not finetuned for chat or Q&A. They should be prompted so that the expected answer is the natural continuation of the prompt.

See `example_text_completion.py` for some examples. To illustrate, see the command below to run it with the llama-2-7b model (`nproc_per_node` needs to be set to the `MP` value):

```
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

### Fine-tuned Chat Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, a specific formatting defined in [`chat_completion`](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212)
needs to be followed, including the `INST` and `<<SYS>>` tags, `BOS` and `EOS` tokens, and the whitespaces and breaklines in between (we recommend calling `strip()` on inputs to avoid double-spaces).

You can also deploy additional classifiers for filtering out inputs and outputs that are deemed unsafe. See the llama-recipes repo for [an example](https://github.com/facebookresearch/llama-recipes/blob/main/inference/inference.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using llama-2-7b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

Llama 2 is a new technology that carries potential risks with use. Testing conducted to date has not — and could not — cover all scenarios.
In order to help developers address these risks, we have created the [Responsible Use Guide](Responsible-Use-Guide.pdf). More details can be found in our research paper as well.

## Issues

Please report any software “bug,” or other problems with the models through one of the following means:
- Reporting issues with the model: [github.com/facebookresearch/llama](http://github.com/facebookresearch/llama)
- Reporting risky content generated by the model: [developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- Reporting bugs and security concerns: [facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md).

## License

Our model and weights are licensed for both researchers and commercial entities, upholding the principles of openness. Our mission is to empower individuals, and industry through this opportunity, while fostering an environment of discovery and ethical AI advancements. 

See the [LICENSE](LICENSE) file, as well as our accompanying [Acceptable Use Policy](USE_POLICY.md)

## References

1. [Research Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
2. [Llama 2 technical overview](https://ai.meta.com/resources/models-and-libraries/llama)
3. [Open Innovation AI Research Community](https://ai.meta.com/llama/open-innovation-ai-research-community/)

For common questions, the FAQ can be found [here](https://github.com/facebookresearch/llama/blob/main/FAQ.md) which will be kept up to date over time as new questions arise. 

## Original LLaMA
The repo for the original llama release is in the [`llama_v1`](https://github.com/facebookresearch/llama/tree/llama_v1) branch.
