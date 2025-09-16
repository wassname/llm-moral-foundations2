import torch
from typing import List, Optional, Any, Dict
from jaxtyping import Float, Int, Bool
from torch import nn, Tensor, functional as F
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer
import pandas as pd
from tqdm.auto import tqdm
from transformers.generation.utils import MinPLogitsWarper, LogitNormalization, RepetitionPenaltyLogitsProcessor
from loguru import logger


from llm_moral_foundations2.gather.choice_tokens import get_choice_tokens_with_prefix_and_suffix, get_special_and_added_tokens, convert_tokens_to_longs
from llm_moral_foundations2.hf import clone_dynamic_cache, symlog

@torch.no_grad()
def force_forked_choice(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    choice_ids: List[List[int]],
    attention_mask: Optional[Int[Tensor, "b s"]] = None,
    forcing_text="\n\nchoice: ",
    unthink_s = "</think>",
    kv_cache: Optional[DynamicCache] = None,
    think=False,
    verbose=False,
    **kwargs
) -> Float[Tensor, "b c"]:
    """
    Force the model to produce a logprob distribution over choices by modifying the input.
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
    attn_k_shape = kv_cache.layers[0].values.shape
    bs = attn_k_shape[0]

    
    input_ids = tokenizer.encode(forcing_text, return_tensors="pt", add_special_tokens=False).to(model.device).repeat((bs, 1))

    # note that when using kv_cache we do not need paste inputs,  but we do need paste attention mask
    if attention_mask is not None:
        new_attn_mask = torch.ones_like(input_ids).long()
        attention_mask = torch.cat([attention_mask, new_attn_mask], dim=1)


    # I need to handle a batch of which some are thinking, some are not. Ideally by masking out this prefix
    
    unthink_ids = tokenizer.encode(unthink_s, return_tensors="pt", add_special_tokens=False).to(model.device).repeat((bs, 1)) 
    if attention_mask is None:
        # Note attentions mask need to cover the cache, and inputs
        attention_mask = torch.ones((bs, attn_k_shape[2] + input_ids.shape[1]), dtype=torch.long, device=model.device)

    # always insert unthink, but sometimes mask it
    input_ids = torch.concat([unthink_ids, input_ids], dim=1)
    attention_mask = torch.cat([torch.ones_like(unthink_ids).long() * think, attention_mask], dim=1)

    # cache_position = attn_k_shape

    o = model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True, past_key_values=kv_cache, use_cache=True, 
        # cache_position=cache_position,
          **kwargs
    )
    logprobs = o.logits[:, -1].log_softmax(dim=-1).float()

    if verbose:
        bi = 0
        # Also print top 10 tokens so I can debug low prob mass
        top_k = logprobs.topk(10, dim=-1)
        print(f"Top 10 tokens for batch {bi} after forcing:")
        print(f"Forcing text: `{forcing_text}`")
        print(f"Input IDs: `{tokenizer.decode(input_ids[bi], skip_special_tokens=False)}`")
        attn_mask_input = attention_mask[bi, -input_ids.shape[1]:]
        print(f"Input IDs: `{tokenizer.decode(input_ids[bi]*attn_mask_input, skip_special_tokens=False)}`")
        for token_id, prob in zip(top_k.indices[bi], top_k.values[bi]):
            print(f"Token: `{tokenizer.decode([token_id])}`, Logprob: {prob.item()}")

        print(f"KV cache size {attn_k_shape}")

        print("-" * 80)

    if choice_ids is None:
        # return all logprobs
        return logprobs

    choice_lprobs = torch.ones(bs, len(choice_ids)) * -1000
    for i, choice_group in enumerate(choice_ids):
        # wait
        choice_group_lprobs = logprobs[:, choice_group]
        choice_lprobs[:, i] = torch.logsumexp(choice_group_lprobs, dim=-1).detach().cpu()

    return choice_lprobs


def get_last_token_id_pos(all_input_ids: Int[Tensor, "s"], token_id, tokenizer) -> int:
    pos = torch.argwhere(all_input_ids == token_id)
    last_pos = pos.max() if len(pos) > 0 else -1
    return last_pos

# def is_thinking(all_input_ids: Int[Tensor, "b s"], tokenizer) -> Bool[Tensor, "b"]:
#     """Check if each sequence is currently thinking"""
#     unthink_token_id = tokenizer.convert_tokens_to_ids("</think>")
#     think_token_id = tokenizer.convert_tokens_to_ids("<think>")
    
#     # HACK: Always assume thinking if we can't determine state
#     # This errs on side of adding </think> when uncertain
#     seq_len = all_input_ids.shape[1]
#
#     # Find last positions using your reverse+argmax trick
#     think_mask = (all_input_ids == think_token_id).flip(dims=[1])
#     unthink_mask = (all_input_ids == unthink_token_id).flip(dims=[1])
    
#     # Handle "not found" case properly
#     has_think = think_mask.any(dim=1)
#     has_unthink = unthink_mask.any(dim=1)
    
#     last_think_pos = torch.where(has_think, seq_len - 1 - think_mask.argmax(dim=1), torch.tensor(-1))
#     last_unthink_pos = torch.where(has_unthink, seq_len - 1 - unthink_mask.argmax(dim=1), torch.tensor(-1))
    
#     return last_think_pos > last_unthink_pos


def is_thinking(
    all_input_ids: Int[Tensor, "b s"],
    tokenizer: PreTrainedTokenizer,
) -> Bool[Tensor, "b"]:
    """Batched check if in thinking state."""
    if all_input_ids.shape[1] == 0:
        return torch.ones(all_input_ids.shape[0], dtype=torch.bool, device=all_input_ids.device)
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    unthink_id = tokenizer.convert_tokens_to_ids("</think>")

    rev_ids = all_input_ids.flip(dims=[1])
    seq_len = all_input_ids.shape[1]

    think_mask = (rev_ids == think_id).float()
    unthink_mask = (rev_ids == unthink_id).float()

    last_think_pos = torch.where(think_mask.any(1), seq_len - 1 - think_mask.argmax(1), torch.full_like(think_mask[:, 0], -1))
    last_unthink_pos = torch.where(unthink_mask.any(1), seq_len - 1 - unthink_mask.argmax(1), torch.full_like(unthink_mask[:, 0], -1))

    # TODO should be bool tensor
    is_thinking = last_think_pos > last_unthink_pos

    return is_thinking

def gen_reasoning_trace_guided(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: Tensor,
    verbose=False,
    attn_mask: Optional[Tensor] = None,
    max_new_tokens: int = 130,
    forcing_text: str = "\n\nchoice:",
    max_thinking_tokens: Optional[int] = None,
    fork_every: int = 10,
    banned_token_ids: Optional[Int[Tensor, "d"]] = None,
    choice_token_ids: Optional[Int[Tensor, "c"]] = None,
    do_sample=False,
    end_think_s = "</think>",
):
    """
    A modified generate that will
    - stop thinking after N tokens
    - fork the generation process and force and answer (cached) every `fork_every` steps
    - avoid banned tokens (by default all special tokens including </think>)
    """
    if banned_token_ids is None:
        banned_token_ids = get_special_and_added_tokens(tokenizer)

    if max_thinking_tokens is not None:
        # add </think> and <think> to banned tokens if not in there
        banned_token_ids += tokenizer.convert_tokens_to_ids(["</think>", "<think>"])
        banned_token_ids = list(set(banned_token_ids))

    all_input_ids = input_ids.clone()

    input_ids = input_ids.to(model.device)

    if verbose:
        inputs_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print("-" * 20 + "inputs" + "-" * 20)
        print(inputs_decoded)
        print("-" * 80)

    bs = input_ids.shape[0]
    nb_input_tokens = input_ids.shape[1]
    data = [[] for _ in range(bs)]

    kv_cache = DynamicCache()

    for i in tqdm(range(max_new_tokens), disable=not verbose):
        o = model.forward(
            input_ids=input_ids, attention_mask=attn_mask, return_dict=True, past_key_values=kv_cache, use_cache=True
        )

        # now we want to modify input so we use cache and newly generated token in the next step
        kv_cache = o.past_key_values

        # Greedy sample
        # FIXME option to use topk or similar
        
        logits_processors = [
            MinPLogitsWarper(min_p=0.1),
            RepetitionPenaltyLogitsProcessor(1.1),
            LogitNormalization()
        ]
        
        logits = o.logits[:, -1].clone()
        logits[:, banned_token_ids] = -float("inf")
        if len(data)<max_thinking_tokens:
            eot_token_id = tokenizer.convert_tokens_to_ids(["</think>"])
            logits[:, eot_token_id] = -float("inf")
        for proc in logits_processors:
            logits = proc(input_ids, logits)

        logp = logits.log_softmax(dim=-1)
        
        if do_sample:
            new_token_id = torch.multinomial(logp.exp(), num_samples=1)
        else:
            new_token_id = logp.argmax(dim=-1, keepdim=True)#.unsqueeze(1)

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

        if is_choice_token or (i % fork_every == 0):
            logp_choices = force_forked_choice(
                model,
                tokenizer,
                # input_ids,
                attention_mask=attn_mask,
                kv_cache=kv_cache,
                think=i>max_thinking_tokens, # always add </think> anyway
                # verbose=i in [5, max_new_tokens // 2 + 5],
                choice_ids=choice_token_ids,
                verbose=verbose,
                forcing_text=forcing_text
            )
            # if probmass.mean()<0.75:
            #     logger.warning(f"Low probability mass detected: {probmass.mean():.2f} at token position {i}. Check your model's interaction with prompt, and choice tokens, and forcing text.")
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

        if (max_thinking_tokens is not None) and (i == max_thinking_tokens):

            # end thinking
            new_input_ids = tokenizer.encode(end_think_s, return_tensors="pt", add_special_tokens=False).to(input_ids.device).repeat((bs, 1))
            input_ids = torch.cat([input_ids, new_input_ids], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.ones_like(new_input_ids).long()], dim=1)

            new_strs = tokenizer.batch_decode(new_input_ids[0], skip_special_tokens=False)
            for j in range(bs):
                for k in range(len(new_strs)):
                    data[j].append(
                        {
                            "token": new_strs[k],
                            "ii": i + 0.5,
                        }
                )

        all_input_ids = torch.cat([all_input_ids, input_ids], dim=1)
        
        # # stop once all samples in the batch has produced an eos, after the inputs
        # if all_input_ids[:, nb_input_tokens:].eq(tokenizer.eos_token_id).any(dim=1).all():
        #     if all_input_ids.shape[1] >= nb_input_tokens + min_new_tokens:
        #         if (max_thinking_tokens is None) or (i >= max_thinking_tokens):
        #             # TODO replace eos, with </think> if it's trying to end without stopping thinking
        #             break
        

    full_strings = tokenizer.batch_decode(all_input_ids, skip_special_tokens=False)

    token_strs = [tokenizer.batch_decode(i) for i in all_input_ids[:, nb_input_tokens:]]

    # convert to one dataframe for each batch
    dfs = [pd.DataFrame(d) for d in data]

    # TODO I might want to remove everything after a tokenizer.eos_token_id

    for i, df in enumerate(dfs):
        df['token_strs'] = token_strs[i]
        df.attrs.update({
            "max_new_tokens": max_new_tokens,
            "max_thinking_tokens": max_thinking_tokens,
            "model_name": model.config._name_or_path,
        })
    return dfs, full_strings


def gen_reasoning_trace(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: Tensor,
    verbose=False,
    attn_mask: Optional[Tensor] = None,
    max_new_tokens: int = 130,
    min_new_tokens: int = 1,
    forcing_text: str = "\n\nchoice: ",
    fork_every: int = 10,
    choice_token_ids: Optional[Int[Tensor, "c"]] = None,
    **kwargs
):
    """
    A modified generate that will
    - fork the generation process and force and answer (cached) every `fork_every` steps
    """
    out = model.generate( 
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        use_cache=True,
        return_dict_in_generate=True,
        **kwargs
    )
    bs = input_ids.shape[0]
    kv_cache = out.past_key_values
    out_token_s = [tokenizer.batch_decode(s, skip_special_tokens=False) for s in out.sequences]
    full_strings = tokenizer.batch_decode(out.sequences, skip_special_tokens=False)    


    data = [[] for _ in range(bs)]
    for ti in range(input_ids.shape[1], out.sequences.shape[1]):
        if (ti%fork_every == 0):

            # clone and crop cache
            kv_cache2 = clone_dynamic_cache(kv_cache, crop=ti)
            think = is_thinking(out.sequences[:, :ti], tokenizer)

            logp_choices = force_forked_choice(
                model,
                tokenizer,
                attention_mask=attn_mask,
                kv_cache=kv_cache2,
                think=think,
                choice_ids=choice_token_ids,
                verbose=verbose and (ti in [fork_every, fork_every*2, max_new_tokens-max_new_tokens%fork_every]),
                forcing_text=forcing_text
            )
        else:
            logp_choices = None
        for j in range(bs):
            data[j].append(
                {
                    "token": out_token_s[j][ti],
                    "token_strs": out_token_s[j][ti],
                    "logp_choices": logp_choices[j].numpy() if (logp_choices is not None) else None,
                    "ii": ti,
                }
            )

    
    dfs = [pd.DataFrame(d) for d in data]

    for ti, df in enumerate(dfs):
        df.attrs.update({
            "max_new_tokens": max_new_tokens,
            "model_name": model.config._name_or_path,
        })

    # TODO check mas df_traj['probmass'].max()>0.5
    return dfs, full_strings
