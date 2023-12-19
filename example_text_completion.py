# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import pandas as pd

from llama import Llama
from typing import List, Optional

from prompts import get_prompts

from torch.profiler import profile
from contextlib import nullcontext

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    prompts_file: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    device: Optional[str] = 'cpu',
    load_weights: Optional[bool] = True,
    model_flavor: Optional[str] = 'llama2', # llama2, pipellama2
    profiling: Optional[bool] = False
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        load_weights=load_weights,
        model_flavor=model_flavor
    )
    prompts: List[str] =  get_prompts(prompt_file=prompts_file)
    
    with profile(profile_memory=True, record_shapes=True, with_stack=True) if profiling else nullcontext() as prof:
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
    if prof is not None:
        prof.export_chrome_trace("trace.json")

    output_file = prompts_file.split('.')[0]
    output = []

    for prompt, result in zip(prompts, results):
        output.append([prompt, 
                       result['generation'],
                       result['metrics']['total_generated_tokens'],  
                       result['metrics']['generated_tokens'], 
                       result['metrics']['latency'], 
                       result['metrics']['per-token-latency']*1e3, 
                       result['metrics']['throughput']])
    
    df = pd.DataFrame(output,
                      columns=[
                        'prompt', 
                        'answer', 
                        'total_generated_tokens', 
                        'generated_tokens', 
                        'latency [s]', 
                        'per_token_latency [ms/token]', 
                        'throughput [token/s]',
                    ])
    
    df.to_csv(f"{output_file}.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
