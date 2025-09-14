# Load the scenario suffixes
# steering
import random
from repeng import DatasetEntry
import json5
import torch
import itertools
from repeng.control import model_layer_list
from repeng import ControlVector, ControlModel, DatasetEntry
from loguru import logger
import contextlib
from anycache import anycache
from llm_moral_foundations2.config import project_dir

@contextlib.contextmanager
def control(model, vector, coeff):
    """
    Usage:
        with control(model, vector, coeff):
            model.generate()
    """
    if coeff==0:
        model.reset()
    else:
        model.set_control(vector, coeff)
    try:
        yield
    finally:
        model.reset()


def wrap_model(model):
    try:
        n_layers = len(model_layer_list(model))
    except Exception as e:
        logger.error(f"Error getting model layers: {e}")
        n_layers = model.config.num_hidden_layers
    # 5 or L//6+2
    # layer_ids = list(range(-4, -n_layers//2, -1)) # halfway to -4
    layer_ids = list(range(-1, -n_layers, -1)) # last layer to first
    cmodel = ControlModel(model, layer_ids)
    return cmodel

@anycache(cachedir='/tmp/anycache.pkl')
def train_steering_vector(model, tokenizer, ds_name="scenario_engagement_dataset", batch_size=2, verbose=False, max_rows=1e9):
    ds_steer = load_steering_ds(tokenizer, ds_name, verbose=verbose)

    if len(ds_steer) > max_rows:
        random.shuffle(ds_steer)
        ds_steer = ds_steer[:max_rows]

    if isinstance(model, ControlModel):
        model.reset()  # make sure you always reset the model before training a new vector

    control_vector = ControlVector.train(
        model,
        tokenizer,
        ds_steer,
        batch_size=batch_size,
        method="pca_diff_weighted",
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

def make_dataset(tokenizer, personas, suffixes, entities, verbose=False, template="You're {persona}, acting in the world."):
    thinking_prefix = get_thinking_prefix(tokenizer)

    # Create dataset entries
    dataset = []
    for i, _suffix in enumerate(suffixes):
        for j, (positive_persona, negative_persona) in enumerate(personas):
            _entity = random.choice(entities)
            positive_persona = positive_persona.format(entity=_entity)
            negative_persona = negative_persona.format(entity=_entity)
            if (thinking_prefix is not None) and random.random() < 0.5:
                suffix = thinking_prefix + _suffix
            else:
                suffix = _suffix + ""

            # tokens = tokenizer.tokenize(suffix, add_special_tokens=False)[:max_suffix_length]
            # Create multiple training examples with different truncations
            # for i in range(1, len(tokens), max(1, len(tokens) // 5)):  # Using stride to reduce dataset size
            positive_prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': template.format(persona=positive_persona)},
                    {'role': 'assistant', 'content': suffix}],
                tokenize=False,
                # enable_thinking=False,
                continue_final_message=True
            )
            negative_prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': template.format(persona=negative_persona)},
                    {'role': 'assistant', 'content': suffix}],
                tokenize=False,
                # enable_thinking=False,
                continue_final_message=True,

            )

            dataset.append(
                DatasetEntry(
                    positive=positive_prompt,
                    negative=negative_prompt
                )
            )

    # shuffle
    random.seed(42)
    random.shuffle(dataset)
    if verbose:
        for i in range(3):
            logger.info(f"Dataset example {i}:\n\npositive_prompt={positive_prompt}\n\nnegative_prompt={negative_prompt}")
    return dataset

def load_entities():
    with open(project_dir/"data/steering/_entities.json5") as f:
        _entities = json5.load(f)
    return _entities

def load_suffixes(collapse=True):
    # loads suffixes
    with open(project_dir/"data/steering/_suffixes.json5") as f:
        suffixes = json5.load(f)
    if collapse:
        suffixes = list(itertools.chain.from_iterable(suffixes.values()))
    return suffixes

def get_thinking_prefix(tokenizer):
    think_token = tokenizer.convert_tokens_to_ids("<think>")
    if think_token:
        return "<think>\n\n"
    else:
        return ""

def load_personas(name):
    with open(project_dir/f"data/steering/{name}.json5") as f:
        personas = json5.load(f)
    return personas['personas']

def load_steering_ds(tokenizer, ds_name="scenario_engagement_dataset", verbose=False):
    with open(project_dir/f"data/steering/{ds_name}.json5") as f:
        scenario_data = json5.load(f) 

    entities = load_entities()
    suffixes = load_suffixes()

    # suffixes = scenario_data["suffixes"]
    personas = scenario_data["personas"]
    dataset = make_dataset(tokenizer, personas, suffixes, entities, verbose=verbose)

    return dataset
