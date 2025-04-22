# Load the scenario suffixes
# steering
from repeng import DatasetEntry
import json
import torch
from repeng.control import model_layer_list
from repeng import ControlVector, ControlModel, DatasetEntry
from loguru import logger
from anycache import anycache
from llm_moral_foundations2.config import project_dir

def wrap_model(model):
    L = len(model_layer_list(model))
    # 5 or L//6+2
    cmodel = ControlModel(model, list(range(-4, -L//2, -1)))
    return cmodel

@anycache(cachedir='/tmp/anycache.pkl')
def train_steering_vector(cmodel, tokenizer, ds_name="scenario_engagement_dataset"):
    ds_steer = load_steering_ds(tokenizer, ds_name)
    cmodel.reset()  # make sure you always reset the model before training a new vector
    control_vector = ControlVector.train(
        cmodel,
        tokenizer,
        ds_steer,
    )
    return control_vector

def find_last_non_whitespace_token(tokenizer, tokens):
    """
    Find the last non-whitespace token in a list of tokens.
    """
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i]
        s = tokenizer.decode(t)
        if len(s.strip()) > 0:
            return t
    return t


def load_steering_ds(tokenizer, ds_name="scenario_engagement_dataset"):
    with open(project_dir/f"data/steering/{ds_name}.json") as f:
        scenario_data = json.load(f)


    # Create dataset entries
    dataset = []
    for suffix in scenario_data["suffixes"]:
        for positive_persona, negative_persona in scenario_data["personas"]:
        
            tokens = tokenizer.tokenize(suffix)
            
            # Create multiple training examples with different truncations
            # We always keep at least 5 tokens at the end for the model to complete
            for i in range(1, len(tokens) - 5, max(1, len(tokens) // 10)):  # Using stride to reduce dataset size
                truncated = tokenizer.convert_tokens_to_string(tokens[:i])

                # TODO use tokenizer formatter instead 
                positive_prompt = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': f"You're a {positive_persona}."},
                     {'role': 'assistant', 'content': truncated}],
                    tokenize=False
                )
                negative_prompt = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': f"You're a {negative_persona}."},
                     {'role': 'assistant', 'content': truncated}],
                    tokenize=False
                )
                
                dataset.append(
                    DatasetEntry(
                        positive=positive_prompt,
                        negative=negative_prompt
                    )
                )

    return dataset
