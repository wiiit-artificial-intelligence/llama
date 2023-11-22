import yaml
import pandas as pd


def get_prompts(
    prompt_file: str
):
    """
    Read prompts from yml file.

    Args:
        prompt_file (str): The path to the prompt file.
    """
    with open(prompt_file, 'r') as file:
        prompts = yaml.safe_load(file)

    return prompts