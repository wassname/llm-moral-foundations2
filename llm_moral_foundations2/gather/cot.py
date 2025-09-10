import torch
from typing import List, Optional, Any, Dict
from jaxtyping import Float, Int
from torch import nn, Tensor, functional as F
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer
import pandas as pd
from llm_moral_foundations2.gather.choice_tokens import get_choice_tokens_with_prefix_and_suffix, get_special_and_added_tokens, convert_tokens_to_longs
from llm_moral_foundations2.hf import clone_dynamic_cache, symlog

@torch.no_grad()
def force_forked_choice(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    # inputs: Int[Tensor, "b s"],
    choice_ids: List[List[int]],
    attention_mask: Optional[Int[Tensor, "b s"]] = None,
    forcing_text="\n\nchoice:",
    kv_cache: Optional[DynamicCache] = None,
    think=False,
    verbose=False,
) -> Float[Tensor, "b c"]:
    """
    Force the model to produce a specific rating by modifying the input.
    This uses a cloned kv_cache so it can fork from a generation process
    Args:
    - think: Whether to exit thinking
    - choices ids: Tensor of token_ids, limited options for the model to output logprobs of
    - forcing text: The text to use to force the model's output, shorter is better
    - inputs: model inputs
    """

    if kv_cache is not None:
        kv_cache = clone_dynamic_cache(kv_cache)

    # modify inputs to force rating
    s = forcing_text

    # might not be needed in thinking only models
    if think:
        s = "</think>" + s

    # bs = kv_cache.key_cache[0].shape[0]
    bs = kv_cache.layers[0].values.shape[0]
    
    input_ids = tokenizer.encode(s, return_tensors="pt", add_special_tokens=False).to(model.device).repeat((bs, 1))

    # note that when using kv_cache we do not need paste inputs,  but we do need paste attention mask
    if attention_mask is not None:
        new_attn_mask = torch.ones_like(input_ids).long()
        attention_mask = torch.cat([attention_mask, new_attn_mask], dim=1)

    o = model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True, past_key_values=kv_cache, use_cache=True
    )
    logprobs = o.logits[:, -1].log_softmax(dim=-1).float()

    if verbose:
        bi = 0
        # print("-" * 20 + "force rating outputs" + "-" * 20)
        # out_string = tokenizer.decode(o.logits.argmax(dim=-1)[bi], skip_special_tokens=True)#[-1]
        # print("decode(outputs)", out_string)
        # print("-" * 80)

        # Also print top 10 tokens so I can debug low prob mass
        top_k = logprobs.topk(10, dim=-1)
        print(f"Top 10 tokens for batch {bi} after forcing:")
        print(f"Forcing text: `{forcing_text}`")
        for token_id, prob in zip(top_k.indices[bi], top_k.values[bi]):
            print(f"Token: {tokenizer.decode([token_id])}, Logprob: {prob.item()}")
        print("-" * 80)

    if choice_ids is None:
        # return all logprobs
        return logprobs

    choice_lprobs = torch.ones(bs, len(choice_ids)) * -1000
    for i, choice_group in enumerate(choice_ids):
        # wait
        choice_group_lprobs = logprobs[:, choice_group]
        choice_lprobs[:, i] = torch.logsumexp(choice_group_lprobs, dim=-1).detach().cpu()

    # choice_lprobs = torch.stack([logprobs[:, i] for i in choice_ids], dim=-1).detach().cpu()
    return choice_lprobs



def gen_reasoning_trace(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    # messages: List[Dict[str, str]],
    input_ids: Tensor,
    device,
    verbose=False,
    attn_mask: Optional[Tensor] = None,
    max_new_tokens: int = 130,
    max_thinking_tokens: int = 125,
    fork_every: int = 10,
    banned_token_ids: Optional[Int[Tensor, "d"]] = None,
    choice_token_ids: Optional[Int[Tensor, "c"]] = None,
):
    """
    A modified generate that will
    - stop thinking half way through
    - fork the generation process and force and answer (cached) every `fork_every` steps
    - avoid banned tokens (by default all special tokens including </think>)
    """
    if banned_token_ids is None:
        banned_token_ids = get_special_and_added_tokens(tokenizer)

    all_input_ids = input_ids.clone()

    input_ids = input_ids.to(device)

    if verbose:
        inputs_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print("-" * 20 + "inputs" + "-" * 20)
        print(inputs_decoded)
        print("-" * 80)

    bs = input_ids.shape[0]
    data = [[] for _ in range(bs)]

    kv_cache = DynamicCache()

    for i in range(max_new_tokens):
        o = model.forward(
            input_ids=input_ids, attention_mask=attn_mask, return_dict=True, past_key_values=kv_cache, use_cache=True
        )

        # now we want to modify input so we use cache and newly generated token in the next step
        kv_cache = o.past_key_values

        # Greedy sample
        logits = o.logits[:, -1].clone()
        logits[:, banned_token_ids] = -float("inf")
        new_token_id = logits.log_softmax(dim=-1).argmax(dim=-1).unsqueeze(1)

        input_ids = new_token_id
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.ones_like(new_token_id).long()], dim=1)

        # check if any of the new tokens, are in the choice_token_ids, if so force answer
        is_choice_token = False
        for bi in range(bs):
            for j in range(len(choice_token_ids)):
                if new_token_id[bi].item() in choice_token_ids[j]:
                    is_choice_token = True
                    break

        if is_choice_token or (i % fork_every == 0) or (i == max_thinking_tokens) or (i > max_thinking_tokens):
            logp_choices = force_forked_choice(
                model,
                tokenizer,
                # input_ids,
                attention_mask=attn_mask,
                kv_cache=kv_cache,
                think=i < max_thinking_tokens,
                # verbose=i in [5, max_new_tokens // 2 + 5],
                choice_ids=choice_token_ids,
                verbose=verbose,
            )
        else:
            logp_choices = None

        new_token = tokenizer.convert_ids_to_tokens(new_token_id)
        for j in range(bs):
            data[j].append(
                {
                    "token": new_token[j],
                    "logp_choices": logp_choices[j].numpy() if logp_choices is not None else None,
                    "ii": i,
                }
            )

        if i == max_thinking_tokens:
            # end thinking
            think_token_id = convert_tokens_to_longs("</think>", tokenizer).to(input_ids.device).repeat((input_ids.shape[0], 1))
            input_ids = torch.cat([input_ids, think_token_id], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.ones_like(think_token_id).long()], dim=1)
            for j in range(bs):
                data[j].append(
                    {
                        "token": "</think>",
                        "ii": i + 0.5,
                    }
                )

        all_input_ids = torch.cat([all_input_ids, input_ids], dim=1)

    full_strings = tokenizer.batch_decode(all_input_ids, skip_special_tokens=False)

    # convert to one dataframe for each batch
    dfs = [pd.DataFrame(d) for d in data]

    return dfs, full_strings
