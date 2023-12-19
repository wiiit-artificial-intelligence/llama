# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

import llama
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from torch.profiler import profile

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
        device: Optional[str] = 'cpu',
        load_weights: Optional[bool] = True,
        model_flavor: Optional[str] = 'llama2' # llama2, pipellama2
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        local_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        backend='gloo'
        if device=='cuda':
            backend='nccl'

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend,
                                                 init_method=f"tcp://{master_addr}:{master_port}",
                                                 rank=local_rank,
                                                 world_size=world_size)

        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ["WORLD_SIZE"])
            initialize_model_parallel(model_parallel_size)

        if device == 'cuda':
            torch.cuda.set_device(local_rank)
        else:
            torch.device(device)

        # seed must be the same in all processes
        torch.manual_seed(seed)        

        # Load model's weights from checkpoint(s)
        start_time = time.time()

        if load_weights:
            checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
            assert model_parallel_size == len(
                checkpoints
            ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
            ckpt_path = checkpoints[get_model_parallel_rank()]
            
            print(f"local-rank: {local_rank} - {ckpt_path} - {get_model_parallel_rank()}")
            checkpoint = torch.load(ckpt_path, map_location=torch.device(device=device))
            
        print(f"Inference will use: {torch.get_num_threads()} cores per worker")

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # Tokenizer
        tokenizer = Tokenizer(model_path=tokenizer_path)

        

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            # Set-up FP32 for more than 1 worker/node/VM cause GLOO does not suport FP16.
            # For one worker we set FP16 as default type before load checpoints due memory constrains.
            if world_size > 1:
                torch.set_default_dtype(torch.float32)
            else:
                # torch.set_default_dtype(torch.bfloat16)
                torch.set_default_dtype(torch.float32)

        # model selection and building
        if model_flavor is None or model_flavor == 'llama2':
            model_args: ModelArgs = llama.model.ModelArgs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                device=device,
                **params,
            )
            model_args.vocab_size = tokenizer.n_words

            model = llama.model.Transformer(model_args)

        elif model_flavor == 'pipellama2':
            model_args: ModelArgs = llama.pipemodel.ModelArgs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                device=device,
                **params,
            )
            model_args.vocab_size = tokenizer.n_words

            model = llama.pipemodel.Transformer(model_args)
        else:
            raise ValueError(f"Model flavor {model_flavor} either not supported or implmented!")

        if device == 'cpu' and world_size == 1:
            torch.set_default_dtype(torch.float32)
        
        if load_weights:
            model.load_state_dict(checkpoint, strict=False)
            print(f"Shard: {ckpt_path} loaded in {time.time() - start_time:.2f} seconds")
        else:
            print("WARNING: Model uninitialized!")

        return Llama(model, tokenizer, device)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device=device

    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        with torch.no_grad():
            params = self.model.params
            device = self.device
            start_time = time.time()
            bsz = len(prompt_tokens)
            print(f"Performing inference in {device} with batch size: {bsz}")
            assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

            min_prompt_len = min(len(t) for t in prompt_tokens)
            max_prompt_len = max(len(t) for t in prompt_tokens)
            assert max_prompt_len <= params.max_seq_len
            total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

            pad_id = self.tokenizer.pad_id
            tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
            for k, t in enumerate(prompt_tokens):
                tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            if logprobs:
                token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

            # Uncomment if you wanna profile inference with Pytorch
            # 
            # with profile(record_shapes=True, 
            #             profile_memory=True,
            #             with_flops=True) as prof:
            prev_pos = 0
            eos_reached = torch.tensor([False] * bsz, device=device)
            input_text_mask = tokens != pad_id
            if min_prompt_len == total_len:
                logits = self.model.forward(tokens, prev_pos)
                token_logprobs = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens,
                    reduction="none",
                    ignore_index=pad_id,
                )

            for cur_pos in range(min_prompt_len, total_len):
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)
                # only replace token if prompt has already been generated
                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token
                if logprobs:
                    token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                        input=logits.transpose(1, 2),
                        target=tokens[:, prev_pos + 1 : cur_pos + 1],
                        reduction="none",
                        ignore_index=pad_id,
                    )
                eos_reached |= (~input_text_mask[:, cur_pos]) & (
                    next_token == self.tokenizer.eos_id
                )
                prev_pos = cur_pos
                if all(eos_reached):
                    break

            # prof.export_chrome_trace(f"model_inference.json")

            if logprobs:
                token_logprobs = token_logprobs.tolist()
            out_tokens, out_logprobs = [], []
            for i, toks in enumerate(tokens.tolist()):
                # cut to max gen len
                start = 0 if echo else len(prompt_tokens[i])
                toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
                probs = None
                if logprobs:
                    probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
                # cut to eos tok if any
                if self.tokenizer.eos_id in toks:
                    eos_idx = toks.index(self.tokenizer.eos_id)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                out_tokens.append(toks)
                out_logprobs.append(probs)

            latency = time.time() - start_time
            gen_toks = [len(L) for L in out_tokens]
            total_gen_toks = sum(gen_toks)
            per_token_latency = latency/total_gen_toks
            throughput = total_gen_toks/latency

            print()
            print("------ Inference metrics ------")
            print(f"Generated tokens: {total_gen_toks}")
            print(f"Latency: {latency:.2f} (s).")
            print(f"Per-token latency: {per_token_latency*1e3:.2f} (ms/token)")
            print(f"Throughput: {throughput:.2f} (tokens/s)")
            print("-------------------------------")

            metrics = {
                "total_generated_tokens": total_gen_toks, 
                "generated_tokens": gen_toks, 
                "latency": latency, 
                "per-token-latency": per_token_latency, 
                "throughput": throughput,
            }

        return (out_tokens, out_logprobs if logprobs else None, metrics)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs, generation_metrics = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t), "metrics": generation_metrics} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs, generation_metrics = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                    "metrics": generation_metrics,
                }
                for t, logprobs_i, metrics, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                },
                "metrics": generation_metrics,
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
