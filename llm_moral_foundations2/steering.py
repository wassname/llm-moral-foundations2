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
    try:
        n_layers = len(model_layer_list(model))
    except Exception as e:
        logger.error(f"Error getting model layers: {e}")
        n_layers = 0
    # 5 or L//6+2
    layer_ids = list(range(-4, -n_layers//2, -1)) # halfway to -4
    layer_ids = list(range(-1, -model.config.num_hidden_layers, -1)) # last layer to first
    cmodel = ControlModel(model, layer_ids)
    return cmodel

@anycache(cachedir='/tmp/anycache.pkl')
def train_steering_vector(model, tokenizer, ds_name="scenario_engagement_dataset", batch_size=2, verbose=False):
    ds_steer = load_steering_ds(tokenizer, ds_name, verbose=verbose)

    # # randomly take 1000
    # ds_steer = ds_steer.shuffle(seed=42).select(range(min(1000, len(ds_steer))))
    if isinstance(model, ControlModel):
        model.reset()  # make sure you always reset the model before training a new vector

    control_vector = ControlVector.train(
        model,
        tokenizer,
        ds_steer,
        batch_size=batch_size,
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

def make_dataset(tokenizer, personas, suffixes, max_suffix_length=10, verbose=False):

    # Create dataset entries
    dataset = []
    for i, _suffix in enumerate(suffixes):
        for j, (positive_persona, negative_persona) in enumerate(personas):
            # tokens = tokenizer.tokenize(suffix, add_special_tokens=False)[:max_suffix_length]
            suffix = _suffix + ""

            # Create multiple training examples with different truncations
            # for i in range(1, len(tokens), max(1, len(tokens) // 5)):  # Using stride to reduce dataset size
            for think in [-1, 0, 1]:
                # truncated = tokenizer.convert_tokens_to_string(tokens[i:])
                match think:
                    case -1:
                        suffix = "<think>\n" + suffix
                    case 1:
                        suffix = "<think>\n\n</think>\n\n" + suffix

                positive_prompt = tokenizer.apply_chat_template(
                    #  f"Please talk about {persona}."
                    # f"Pretend you're an {persona} person making statements about the world. 
                    # "Act as if you're extremely {persona}.",
                    [{'role': 'user', 'content': f"You're a {positive_persona}."},
                        {'role': 'assistant', 'content': suffix}],
                    tokenize=False,
                    enable_thinking=False,
                    continue_final_message=True
                )
                negative_prompt = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': f"You're a {negative_persona}."},
                        {'role': 'assistant', 'content': suffix}],
                    tokenize=False,
                    enable_thinking=False,
                    continue_final_message=True,

                )
                if verbose and (i == 0) and (j == 0):
                    logger.info(f"Detokenized: {positive_prompt}")

                dataset.append(
                    DatasetEntry(
                        positive=positive_prompt,
                        negative=negative_prompt
                    )
                )
    return dataset


def load_steering_ds(tokenizer, ds_name="scenario_engagement_dataset", verbose=False):
    with open(project_dir/f"data/steering/{ds_name}.json") as f:
        scenario_data = json.load(f) 

    suffixes = scenario_data["suffixes"]
    personas = scenario_data["personas"]
    dataset = make_dataset(tokenizer, personas, suffixes, verbose=verbose)

    return dataset
