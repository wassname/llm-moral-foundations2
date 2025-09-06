from loguru import logger
import torch
import re
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_kwargs, device="auto"):
    """
    Hopefully modelkwargs can handle
    - bfloat16 but now with awq
    - quantization, bnb?
    - lora, peft, etc not supported yet


    ref 
    - https://github.com/general-preference/general-preference-model/blob/main/general_preference/models/rw_model_general_preference.py#L17
    - https://github.com/huggingface/trl/blob/4871c82b0cd1caae72522182f9171ea069481250/trl/trainer/utils.py#L877
    https://github.com/unslothai/unsloth/blob/71039cb1ce88034f12855476f2a0c5ff63ad59a7/unsloth/models/_utils.py#L295
    """
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None


    # model_kwargs.pop("params_B")

    if model_kwargs.pop("load_in_4bit", False):
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
    elif model_kwargs.pop("load_in_8bit", False):
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
    tokenizer = load_tokenizer(model_kwargs)
    id = model_kwargs.pop("id")
    model = AutoModelForCausalLM.from_pretrained(
        device_map=device,
        torch_dtype=torch_dtype,
        pretrained_model_name_or_path=id,
        # **model_kwargs,
    )
    return model, tokenizer

def load_tokenizer(model_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_kwargs["id"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def work_out_batch_size(model_kwargs, gpu_mem_gb=24):
    """
    Work out the batch size based on the model size and GPU memory.
    """
    # 1. Get the number of parameters in the model
    params_B = model_kwargs["params_B"]
    # 2. Get the GPU memory in bytes
    gpu_mem_bytes = gpu_mem_gb * 1024**3
    # Quantization
    if model_kwargs.get("load_in_4bit", False):
        params_B /= 4
    elif model_kwargs.get("load_in_8bit", False):
        params_B /= 8
    # 3. Calculate the batch size
    batch_size = int(gpu_mem_bytes / (params_B * 1024**3))
    return batch_size

