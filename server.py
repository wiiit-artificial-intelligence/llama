import subprocess
from flask import Flask, request, jsonify, Response
import os
import sys
import select
import re
import json
import logging
import colorlog
import multiprocessing
import platform
import time

from llama import Llama 
from llama.tokenizer import Tokenizer
from typing import List, Optional
from llama import Llama, Dialog

SERVER_NAME = f"llama-2-pytorch"
SERVER_VERSION = "0.1.0"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_MAX_GEN_LEN = 1024
DEFAULT_SEED = 1
DEFAULT_EARLY_STOP = True
DEFAULT_TASK = "chat"
DEFAULT_STREAM = int(False)
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = 'cuda'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

def initialize_generator(checkpoint_dir, tokenizer_path):
    global generator  # Declare generator as global to access it outside the function
    generator = Llama.build(
        ckpt_dir=checkpoint_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        max_batch_size=DEFAULT_BATCH_SIZE,
        device=DEFAULT_DEVICE,
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

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler for the log file
file_handler = logging.FileHandler('server.log')
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set it to both handlers
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s: %(message)s',
    log_colors={
        'DEBUG': 'reset',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Config server infra
logger.debug(f"{SERVER_NAME} - {SERVER_VERSION}")

# Start Server
app = Flask(__name__)

# read available models
MODELS_PATH = "./models"
model_names = [m for m in os.listdir(MODELS_PATH) if os.path.isdir(os.path.join(MODELS_PATH, m))]

if len(model_names) > 0:
    logger.info("models available: " + ' '.join(model_names))
else:
    logger.warning(f"No models found in {MODELS_PATH}")
    model_names = []

# set default model
DEFAULT_MODEL = "llama-2-7b-chat"
if len(model_names) > 0:
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in model_names else model_names[0]
else:
    default_model = None

ckpt_dir = f"{MODELS_PATH}/{DEFAULT_MODEL}/"
tokenizer_path = f"{MODELS_PATH}/{DEFAULT_MODEL}/tokenizer.model"


# Initialize the generator before starting the server
initialize_generator(checkpoint_dir=ckpt_dir,
                     tokenizer_path=tokenizer_path)


def load_model_schema(model_name):
    file_path = os.path.join(MODELS_PATH, model_name, "model_schema.json")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                schema = json.load(file)
                return schema
            except json.JSONDecodeError as e:
                logger.error(f"Error loading model schema JSON file {file_path}: {e}")
                return None
    else:
        logger.error(f"The model schema JSON file {file_path} does not exist!")
        return None

def inference_request_data(data):
    errors = []
    try:
        # inputs
        if "inputs" in data.keys():
            if not isinstance(data["inputs"], list):
                errors.append("inputs must be a list!")
            elif len(data["inputs"]) == 0:
                errors.append("user_prompts must have on prompt at least!")
            elif len(data["inputs"]) > 1:
                errors.append("This model does not support batched inferences (more than one prompt)!")

            inputs = []
            for _input in data["inputs"]:
                inputs.append({
                    "user_prompts":   _input.get("user_prompts", ""),
                    "system_prompts": _input.get("system_prompts", "")
                })

        else:
            errors.append("Missing inputs key!")
        
        # parameteres
        if "parameters" in data.keys():
            parameters = {
                "early_stop":   bool(data["parameters"].get("early_stop", DEFAULT_EARLY_STOP)),
                "max_seq_len":   int(data["parameters"].get("max_gen_len", DEFAULT_MAX_GEN_LEN)),
                "max_gen_len":   int(data["parameters"].get("max_gen_len", DEFAULT_MAX_GEN_LEN)),
                "task":          str(data["parameters"].get("task", DEFAULT_TASK)),
                "temperature": float(data["parameters"].get("temperature", DEFAULT_TEMPERATURE)),
                "top_p":       float(data["parameters"].get("top_p", DEFAULT_TOP_P)),
                "seed":          int(data["parameters"].get("seed", DEFAULT_SEED)),
                "stream":       bool(data["parameters"].get("stream", DEFAULT_STREAM))
            }
        else:
            # default parameters
            parameters = {
                "early_stop": DEFAULT_EARLY_STOP,
                "max_seq_len": DEFAULT_MAX_SEQ_LEN,
                "max_gen_len": DEFAULT_MAX_GEN_LEN,
                "task": DEFAULT_TASK,
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
                "seed": DEFAULT_SEED,
                "stream": DEFAULT_STREAM
            }
        
    except Exception as e:
        logger.error(e)
        errors.append(e)

    return inputs, parameters, errors

# API V2 - server - metadata
@app.route('/v2', methods=['GET'])
def get_server_metedata():
    response = {
        "name": SERVER_NAME,
        "versions": SERVER_VERSION,
        "extensions": [""],
        "models": model_names
    }
    return jsonify(response), 200

# API V2 - server health - live
@app.route('/v2/health/live', methods=['GET'])
def check_health_live():
    return jsonify({'message': 'OK'}), 200

# API V2 - server health - ready
@app.route('/v2/health/ready', methods=['GET'])
def check_health_ready():
    if len(model_names) > 0:
        return jsonify({'message': 'OK'}), 200
    else:
        return jsonify({'error': 'Not models avaliable!'}), 404 

# API V2 - models - metadata
@app.route('/v2/models/<model_name>', methods=['GET'])
def get_model_metedata(model_name):
    if model_name in model_names:    
        response = {
            "name": model_name,
            "versions": [],
            "platform": "",
        }
        # append model schema
        model_schema = load_model_schema(model_name)
        if model_schema is not None:
            response.update(model_schema)

        return jsonify(response), 200
    else:
        return jsonify({'error': 'Model not found!'}), 404 

# API V2 - models - ready
@app.route('/v2/models/<model_name>/ready', methods=['GET'])
def check_model_ready(model_name):
    if model_name in model_names:
        return jsonify({'message': 'OK'}), 200
    else:
        return jsonify({'error': 'Model not found!'}), 404 

# API V2 - models - inference
@app.route('/v2/models/<model_name>/infer', methods=['POST'])
def models_inference(model_name):  
    # get POST body data
    data = request.json
    logger.debug(data)
    data_id = data.get("id", 0)

    # extract inputs and parameter for inference
    inputs, parameters, errors = inference_request_data(data)

    if errors is not None and len(errors) > 0:
        # Internal Server Error
        logger.error(errors)
        return jsonify({'error': errors}), 500
      
    # inference
    if parameters["stream"]:
        # stream the inference result
        logger.info("Running inference (streaming)")

        try:
            # Start the process
            if inputs[0]["user_prompts"] is None:
                return jsonify({'error': 'Missing or invalid input'}), 400
    
            def extract_tokens(prompts):

                prompt_tokens = get_dialog_token(prompts)

                token_stream = []
                decoded_token = ""

                inference_start_time = time.time()
                # enabling yield_token converts generate method into a generator of tokens
                # an iteration is needed to get each token
                tokens_generator = generator.generate_token(
                    prompt_tokens=prompt_tokens,
                    max_gen_len=parameters["max_gen_len"],
                    temperature=parameters["temperature"],
                    top_p=parameters["top_p"],
                    logprobs=False,
                    echo=False)

                token_times = []
                for token, token_time in tokens_generator:
                    token_times.append(token_time)
                    # Decode token                
                    decoded_token = generator.tokenizer.sp_model.id_to_piece(token)[0]
                    decoded_token = decoded_token.replace("▁"," ") 

                    # Yield decoded token               
                    yield decoded_token

                latency = time.time() - inference_start_time

                metrics={
                    "sequence_length": len(prompt_tokens[0]),
                    "forward_passes": len(token_times),
                    "generated_tokens": len(token_times),
                    "latency": latency,
                    "throughput": len(token_times)/latency,
                    "TTFT_ms": token_times[0]*1e3,
                    "TPOT_ms": sum(token_times)/len(token_times)*1e3,
                    "TPOT_us_list": [tok_time * 1e6 for tok_time in token_times],
                }

                metadata = f"\n\n[METADATA]\n{json.dumps(metrics)}\n[/METADATA]\n"

                yield metadata

            return Response(extract_tokens(inputs[0]["user_prompts"]), content_type='text/plain')

        except KeyboardInterrupt:
            # Handle keyboard interrupt (Ctrl+C)
            logger.warning("\nProcess interrupted. Exiting...")
            return jsonify({'error': "Inference interrupted!"}), 500
        except Exception as e:
            logger.error(e)
            return jsonify({'error': str(e)}), 500
    else:
        # wait until inference end and send the result in a single response
        logger.info("Running inference")

        try:
            results = generator.chat_completion(
                dialogs=inputs[0]["user_prompts"],
                max_gen_len=parameters["max_gen_len"],
                temperature=parameters["temperature"],
                top_p=parameters["top_p"],
                early_stop_generation=parameters["early_stop"],
            )

            request_reply = {
                "model_name": model_name,
                "model_version": "",
                "id": data_id,
                "outputs":[
                    {
                        'response': results[0]['generation']['content'],
                        'metadata': results[0]['metrics']
                    }
                ]
            }

            print(request_reply)
            return jsonify(request_reply), 200
        except Exception as e:
            logger.error(e)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, 
            host='0.0.0.0', 
            port=5000)