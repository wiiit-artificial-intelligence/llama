from flask import Flask, request, Response, jsonify
import subprocess

from llama import Llama 
from llama.tokenizer import Tokenizer
from typing import List, Optional
from llama import Llama, Dialog

app = Flask(__name__)

# Initialize the model
ckpt_dir = '../models/llama-2-7b-chat/'
tokenizer_path = '../models/llama-2-7b-chat/tokenizer.model'
max_seq_len = 1024
max_batch_size = 16
device = 'cuda'

default_max_gen_len = 64
default_temperature = 0.0
default_top_p = 0.9


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

PRINT_RESULTS = False

if device == 'cuda':
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
    inference_hardware = result.stdout.strip().split('\n')

def initialize_generator():
    global generator  # Declare generator as global to access it outside the function
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        do_profile=False,
        profile_output='/app/log/test',
        init_method='checkpoint', # checkpoint file, random
        data_type='default',
    )

def get_dialog_token(dialogs):
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
                generator.tokenizer.encode(
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
        dialog_tokens += generator.tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            bos=True,
            eos=False,
        )
        prompt_tokens.append(dialog_tokens)

    return prompt_tokens

# Heartcheck endpoint
@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    return jsonify({'status': 'alive', 'model': ckpt_dir.split('/')[2], 'platform': inference_hardware})

# Generate endpoint
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompts = data['prompts']
    max_gen_len = data.get('max_gen_len', default_max_gen_len)
    temperature = data.get('temperature', default_temperature)
    top_p = data.get('top_p', default_top_p)
    early_stop = data.get('early_stop', True)
    task = data.get('task', 'dialogue')
      
    if task == 'dialogue':
        results = generator.chat_completion(
            dialogs=prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            early_stop_generation=early_stop,
        )
    elif task == 'text':
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            early_stop_generation=early_stop,
        )

    return jsonify(results)

# Generate endpoint for streaming
@app.route('/generate-stream', methods=['POST'])
def chat_stream():
    try:
        # Get the input string from the request body
        data = request.json
        prompts = data['prompts']
        max_gen_len = data.get('max_gen_len', default_max_gen_len)
        temperature = data.get('temperature', default_temperature)
        top_p = data.get('top_p', default_top_p)
        early_stop = data.get('early_stop', True)
        task = data.get('task', 'dialogue')
        
        if prompts is None:
            return jsonify({'error': 'Missing or invalid input'}), 400
 
        def extract_tokens(prompts):
            prompt_tokens = get_dialog_token(prompts)

            token_stream = []
            decoded_token = ""

            # enabling yield_token converts generate method into a generator of tokens
            # an iteration is needed to get each token
            tokens = generator.generate_token(
                            prompt_tokens=prompt_tokens,
                            max_gen_len=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                            logprobs=False,
                            echo=False)

            for token in tokens:
                # Decode token                
                decoded_token = generator.tokenizer.sp_model.id_to_piece(token)[0]

                decoded_token = decoded_token.replace("▁"," ") 

                if decoded_token.startswith("<0x") or decoded_token.startswith("</"):
                    if decoded_token == "<0x0A>" or decoded_token == ("</s>"):
                        decoded_token = generator.tokenizer.sp_model.detokenize(token)
                    else:
                        decoded_token = ""

                # Yield decoded token               
                yield decoded_token

            if PRINT_RESULTS:
                print(result)

        return Response(extract_tokens(prompts), content_type='text/plain')

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

# Initialize the generator before starting the server
initialize_generator()

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)
