import torch
from typing import List, Optional, Any, Dict
from jaxtyping import Float, Int
from torch import nn, Tensor, functional as F
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer


def convert_tokens_to_longs(tokens: List[str], tokenizer: PreTrainedTokenizer):
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if not isinstance(ids, list):
        ids = [ids]
    return torch.LongTensor(ids)


def get_choice_tokens_with_prefix_and_suffix(choices: List[str], tokenizer: PreTrainedTokenizer, prefixes = ["Ġ", " ", "\n", ".", "_"], suffixes = [",", ".", " "]) -> List[int]:
    """
    When we are looking for specific output tokens, they might exist in multiple version e.g. " Yes", "Yes", "Yes ", "\n"Yes" depending on the tokenizer. This attempts to get all combinations
    """
    
    outs = []
    for c in choices:
        token_id = tokenizer.encode(c, return_tensors="pt")[0, -1].item()
        outs.append(token_id)

        for p in prefixes:
            token_id = tokenizer.encode(p + c, return_tensors="pt")[0, -1].item()
            outs.append(token_id)
        for s in suffixes:
            token_id = tokenizer.encode(s + c, return_tensors="pt")[0, -1].item()
            outs.append(token_id)

    # dedup
    outs = list(set(outs))
    # remove None
    outs = [id for id in outs if id is not None]

    # make sure each decodes to something that contains at least one of the choices
    outs2 = []
    for id in outs:
        decoded = tokenizer.decode([id]).strip()
        if any(choice in decoded for choice in choices):
            outs2.append(id)

    return outs2

def get_special_and_added_tokens(tokenizer: PreTrainedTokenizer, verbose=False) -> Optional[Int[Tensor, "banned"]]:
    """Get the special and added tokens so we can ban them in a controlled the generation process."""
    # get all types of special tokens
    additional_special_tokens = tokenizer.special_tokens_map_extended["additional_special_tokens"]
    special_tokens = [i for i in tokenizer.special_tokens_map_extended.values() if isinstance(i, str)]
    added_vocab = tokenizer.get_added_vocab()
    banned_tokens = additional_special_tokens + special_tokens + list(added_vocab.keys())

    # convert to id
    banned_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in banned_tokens]
    banned_token_ids = [i for i in banned_token_ids if i is not None]

    # dedup
    banned_token_ids = torch.LongTensor(list(set(banned_token_ids)))
    if verbose:
        print(tokenizer.batch_decode(banned_token_ids[:, None], skip_special_tokens=False))
    return banned_token_ids


# get_banned_tokens(tokenizer, verbose=True)
