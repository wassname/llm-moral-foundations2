from loguru import logger
import copy
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from transformers import DataCollatorWithPadding
import srsly
import os

from llm_moral_foundations2.load_model import load_model, work_out_batch_size
from llm_moral_foundations2.utils import sanitize_filename
from llm_moral_foundations2.steering import wrap_model, load_steering_ds, train_steering_vector
from llm_moral_foundations2.config import project_dir
from llm_moral_foundations2.data import batch_tokenize

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# models id, billion_parameters
models = [
    # misaligned
    dict(id="drfellx/emergent_misalignment_test_qwen2.5-7B-Instruct", params_B=7, load_in_8bit=True, tags=["qwen", "misaligned"]),
    dict(id="wassname/qwen-2.5-coder-3B-sft-ultrachat-fourchan", params_B=3, load_in_8bit=True, tags=["qwen" "chat"]),

    # notable
    dict(id="NousResearch/Hermes-3-Llama-3.2-3B", params_B=3, load_in_8bit=True, tags=["nous", "chat"]),
    dict(id="unsloth/Qwen2.5-7B-Instruct", params_B=7, load_in_8bit=True, tags=["qwen"]),
    dict(id="microsoft/Phi-4-mini-instruct", params_B=3, load_in_8bit=True, tags=["phi", "chat"]),
    dict(id="unsloth/gemma-2-9b-it", params_B=9, load_in_8bit=True, tags=["gemma", "chat"]),
    dict(id="Meta-Llama-3.1-8B-Instruct", params_B=8, load_in_8bit=True, tags=["llama3", "chat"]),

    # RAG and [Judging](https://verdict.haizelabs.com/), reranking, and forecasting models seem more morally balanced
    dict(id="opencompass/CompassJudger-1-7B-Instruct", params_B=7, load_in_8bit=True, tags=["judge"],),
    dict(id="CohereLabs/c4ai-command-r7b-12-2024", params_B=7, load_in_8bit=True, tags=["rag"]),
    dict(id="allenai/TruthfulQA-Truth-Judge-Llama2-7B", params_B=7, load_in_8bit=True, tags=["judge"]),
    dict(id="OpenSafetyLab/MD-Judge-v0.1", params_B=7, load_in_8bit=True, tags=["judge"]),



    # Code and math?
    # Qwen/Qwen2.5-Math-7B-Instruct
    # Qwen/Qwen2.5-Coder-7B-Instruct

    ## Minor but notable
    dict(id="HuggingFaceH4/mistral-7b-grok", params_B=7, load_in_8bit=True, tags=["grok"]),
    dict(id="HuggingFaceH4/mistral-7b-anthropic", params_B=7, load_in_8bit=True, tags=["grok"]),
    dict(id="HuggingFaceH4/zephyr-7b-alpha", params_B=7, load_in_8bit=True, tags=["h4"]),
    dict(id="allenai/OLMo-2-0325-32B-Instruct", params_B=32, load_in_8bit=True, tags=["allenai"]),
    dict(id="ibm-granite/granite-3.3-8b-instruct", params_B=8, load_in_8bit=True, tags=["granite"]),

    ## QC: right wing models are right wing?
    dict(id="dpasch01/pp-llama3-8b-right-wing", params_B=8, load_in_8bit=True, tags=["llama3", "right-wing"]),
    dict(id="dpasch01/pp-llama3-8b-left-wing", params_B=8, load_in_8bit=True, tags=["llama3", "left-wing"]),

    ## How does uncensoring effect morality?
    dict(id="huihui-ai/DeepSeek-R1-Distill-Qwen-7B-abliterated-v2", params_B=7, load_in_8bit=True, tags=["qwen", "abliterated"]),
    dict(id="nicoboss/DeepSeek-R1-Distill-Qwen-7B-Uncensored-Reasoner", params_B=7, load_in_8bit=True, tags=["qwen", "uncensored"]),
    dict(id="deepseek/DeepSeek-R1-Distill-Qwen-7B", params_B=7, load_in_8bit=True, tags=["qwen",]),
    # "cognitivecomputations/Dolphin3.0-Llama3.2-3B"
    # "cognitivecomputations/Dolphin3.0-Llama3.1-8B"
]

@torch.no_grad()
def process_model(model_kwargs, args):
    max_model_len = args.max_model_len
    output_dir = project_dir / "data" / "eval_results2"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_id = model_kwargs["id"]
    model_id_fn = sanitize_filename(model_id)
    batch_size = work_out_batch_size(model_kwargs, gpu_mem_gb=args.gpu_mem_gb)

    model_kwargs = copy.deepcopy(model_kwargs)

    logger.info(f"Loading model {model_kwargs['id']} bs={batch_size}")
    model, tokenizer = load_model(model_kwargs)

    # steering
    cmodel = wrap_model(model)
    logger.info("Train steering model")
    control_vector = train_steering_vector(cmodel, tokenizer, ds_name=args.steer_name)

    # load data, and permute choices
    data = pd.read_csv(project_dir /'data/mcf/vignettes.csv').reset_index()
    ds = Dataset.from_dict(data)
    ds = ds.map(batch_tokenize, batched=True, fn_kwargs=dict(tokenizer=tokenizer, max_model_len=max_model_len), remove_columns=data.columns.tolist())

    MAX_ROWS = min(args.max_rows, len(ds))
    if MAX_ROWS < len(ds):
        logger.info(f"Truncating dataset to {MAX_ROWS} rows")
        ds = ds.shuffle(seed=42).select(range(MAX_ROWS))

    for steer_v in [-2, 0, 2]:
        f_res = output_dir / f"{model_id_fn}-{MAX_ROWS}-steer{steer_v}.jsonl.gz"
        if f_res.exists():
            logger.info(f"Skipping {f_res} as it already exists")
            continue
        if steer_v == 0:
            cmodel.reset()
        else:
            cmodel.set_control(control_vector, coeff=steer_v)

        dl = DataLoader(ds.with_format("torch"), batch_size=batch_size, collate_fn=DataCollatorWithPadding(tokenizer, padding=True),)

        results = []
        for i, x in enumerate(tqdm(dl, f'eval bs={batch_size} steer_v={steer_v}', total=len(dl), unit="batch")):
            x = x.to(model.device)


            o = model.forward(input_ids=x["input_ids"], attention_mask=x["attention_mask"], return_dict=True)
            logprobs = torch.log_softmax(o.logits, -1)[:, -1, :].cpu().float()  # take last token
            num_strs = [str(x) for x in range(4)]
            num_ids = [x[-1] for x in tokenizer.batch_encode_plus(num_strs)["input_ids"]]

            # get logits, log results
            
            for ri in range(logprobs.shape[0]):
                choice_logprobs_permuted = [logprobs[ri, i].item() for i in num_ids]
                prob_mass = torch.tensor(choice_logprobs_permuted).exp().sum().item()

                r = dict(
                    choice_logprobs_permuted=choice_logprobs_permuted,
                    prob_mass=prob_mass,
                    steer_v=steer_v,
                    index=x["index"][ri].item(),
                    reversed=x["reversed"][ri].item(),
                    steer_name=args.steer_name,
                )
                results.append(r)

                # QC
                if i==0 and ri==0:
                    j = i * batch_size + ri
                    # top logprobs (helps detect other choices the model wants to take and tokenizer errors)
                    top_logprobs = torch.topk(logprobs[ri], 10)
                    top_logprobs = {tokenizer.decode([x], skip_special_tokens=True): y.item() for x, y in zip(top_logprobs.indices, top_logprobs.values)}
                    logger.info(f"Top logprobs\n{top_logprobs}")
                    logger.info(f"Prob mass: {prob_mass}")

                    # gen full answer
                    gen = model.generate(
                        input_ids=x["input_ids"][ri:ri+1],
                        attention_mask=x["attention_mask"][ri:ri+1],
                        max_new_tokens=400, min_new_tokens=300, do_sample=False)
                    s_new = tokenizer.decode(gen, skip_special_tokens=False)
                    logger.info(f"Gen: {s_new}")
                    f_samples = output_dir / f"{model_id_fn}-steer{steer_v}-sample{j}.md"
                    f_samples.write_text(s_new, encoding="utf-8")

        
        srsly.write_gzip_jsonl(f_res, results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=1024)
    parser.add_argument("--max_perms", type=int, default=2)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--gpu_mem_gb", type=int, default=24)
    parser.add_argument("--steer_name", type=str, default="scenario_engagement_dataset")
    args = parser.parse_args()


    for model_kwargs in models:
        process_model(model_kwargs, args)
