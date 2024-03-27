import requests
import re
import os
import json
from pprint import pprint

# select the model to use
MODEL_NAME = "llama-2-7b-chat"

SERVER_URL = "http://127.0.0.1:5000"
ENDPOINT_URL_INFERENCE = f"{SERVER_URL}/v2/models/{MODEL_NAME}/infer"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.0
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_MAX_GEN_LEN = 0
DEFAULT_SEED = 0
DEFAULT_EARLY_STOP = True
DEFAULT_TASK = "chat"
DEFAULT_STREAM = int(False)

payload_chat = {
    "inputs": [
        {
            "user_prompts": ["number of planets in the solar system",
                             "What is your name?"],
            "system_prompts": ["Answer in spanish", 
                               "Answer without using emojis"]
        }
    ],
    "parameters": {
        "early_stop": DEFAULT_EARLY_STOP,
        "max_seq_len": DEFAULT_MAX_SEQ_LEN,
        "max_gen_len": 80,
        "task": "chat",
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "seed": DEFAULT_SEED,
        "stream": DEFAULT_STREAM
    }
}

payload_text = {
    "inputs": [
        {
            "user_prompts": ["I'm a robot that comes from future. My name is"],
        }
    ],
    "parameters": {
        "early_stop": DEFAULT_EARLY_STOP,
        "max_seq_len": DEFAULT_MAX_SEQ_LEN,
        "max_gen_len": 64,
        "task": "generate",
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "seed": DEFAULT_SEED,
        "stream": DEFAULT_STREAM
    }
}

payloads = [payload_chat, payload_text]

pattern = re.compile(r'<0x([0-9A-Fa-f]{2})>')

# Function to replace hexadecimal representations with symbols
def replace_hex(match):
    hex_string = match.group(1)  # Extract the hexadecimal string
    byte_value = int(hex_string, 16)  # Convert hexadecimal string to byte value
    symbol = chr(byte_value)  # Get the symbol corresponding to the byte value
    return symbol


if __name__ == "__main__":
    # check if server is alive
    response = requests.get(SERVER_URL + "/v2/health/live")
    if response.status_code == 200:
        print("Server is alive!")
    else:
        print(f"{response.status_code}: Error: {response.json()['error']}")
    
    # check if server is alive
    response = requests.get(SERVER_URL + "/v2/health/ready")
    if response.status_code == 200:
        print("Server is ready!")
    else:
        print(f"{response.status_code}: Error: {response.json()['error']}")


    for payload in payloads:
        # Streaming request
        print("-"*32 + "\nTesting Streaming\n" + "-"*32)
        print(f">>> {payload['inputs'][0]['user_prompts']}")

        payload["parameters"]["stream"] = 1

        response = requests.post(ENDPOINT_URL_INFERENCE, json=payload, stream=True)
        new_tokens = []

        if response.status_code == 200:
            full_response = ""
            for token in response.iter_content(chunk_size=8):
                if token:
                    try:
                        decoded_token = pattern.sub(replace_hex, token.decode('utf-8'))
                        full_response += decoded_token

                        # printing or not printing
                        new_tokens.append(decoded_token)
                        if not "[METADATA]" in full_response and (decoded_token == '\n' or ' ' in decoded_token):
                            # print
                            print(''.join(new_tokens), end="", flush=True)
                            new_tokens = []
                        # else: wait to print

                    except:
                        pass

            if ("[METADATA]" in full_response and "[/METADATA]" in full_response):
                delimiters = "[METADATA]", "[/METADATA]"
                regex_pattern = '|'.join(map(re.escape, delimiters))
                response_metadata = json.loads(re.split(regex_pattern, full_response)[1])
                print(" ")
                pprint(response_metadata)
        else:
            print(f"Error {response.status_code}: {response.json()['error']}")

        # Non-streaming request
        print("-"*32 + "\nTesting Non-Streaming\n" + "-"*32)
        print(f">>> {payload['inputs'][0]['user_prompts']}")

        payload["parameters"]["stream"] = 0

        response = requests.post(ENDPOINT_URL_INFERENCE, json=payload)

        if response.status_code == 200:
            full_response = response.json()

            for answer in full_response["outputs"]:
                if 'response' in answer:
                    print(answer['response'])
                if 'metadata' in answer: 
                    pprint(answer['metadata'])
        else:
            print(f"Error {response.status_code}: {response.json()['error']}")

