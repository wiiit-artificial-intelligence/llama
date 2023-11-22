# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
import pandas as pd

from llama import Llama, Dialog

from prompts import get_prompts

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    prompts_file: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    device: Optional[str] = 'cpu',
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
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
    )

    dialogs: List[Dialog] = get_prompts(prompt_file=prompts_file)

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    output = []
    output_file = prompts_file.split('.')[0]

    for dialog, result in zip(dialogs, results):

        for msg in dialog:
            prompt = msg['content']
        
        output.append([prompt, 
                            result['generation']['content'],
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
