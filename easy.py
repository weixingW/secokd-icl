#import packages
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from data import *
from pprint import pprint
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass, field
from data.prompt_params import *
import json 
import logging
from tqdm import tqdm
work_dir = os.getcwd()
with open(os.path.join(work_dir,'data/gsm8k.yaml'), 'r') as file:
    GSM8K = yaml.safe_load(file)
with open(os.path.join(work_dir,'data/arc.yaml'), 'r') as file:
    ARC = yaml.safe_load(file)
with open(os.path.join(work_dir,'data/aqua.yaml'), 'r') as file:
    AQUA = yaml.safe_load(file)
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)



available_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {available_device}")

# Define and parse arguments.
@dataclass
class ScriptArguments:

    model_name: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    output_name: Optional[str] = field(default="./results/lora_kd_alpha0.5", metadata={"help": "the output directory"})
    adapter_path: Optional[str] = field(default=None, metadata={"help": "the adapter path"})
    dataset_name: Optional[Union[list,str]] = field(
        default="arc_challenge", metadata={"help": "the dataset name, optionally a list of datasets"}
    )
    device: Optional[str] = field(default=available_device, metadata={"help": "the device to use"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "use peft model"})
    use_8_shot: Optional[bool] = field(default=False, metadata={"help": "use 8-shot setting"})

def format_one_shot_arc(demo, sample, tokenizer):
    prompt = ""
    prompt += f"Question:{demo['question']}\nAnswer: {demo['rational']} The answer is: {demo['labels']}\n\n"
    prompt += f"Question:{sample['question']} {sample['choices']}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"]

def format_8_shot_arc(demo_list, sample, tokenizer):
    prompt = ""
    for demo in demo_list:
        prompt +=f"Question:{demo['question']}\nAnswer: {demo['rational']} The answer is: {demo['labels']}\n\n"
    prompt += f"Question:{sample['question']} {sample['choices']}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"]


def format_one_shot_gsm8k(demo, sample, tokenizer):
    prompt = f"Q:{demo['question']}"
    prompt += f"\nA: {demo['rational']} The answer is #### {demo['answer']}\n\n"
    prompt += f"Q:{sample['question']}\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"].replace(",", "")

def format_8_shot_gsm8k(demo_list, sample, tokenizer):
    prompt = ""
    for demo in demo_list:
        prompt += f"Q:{demo['question']}"
        prompt += f"\nA: {demo['rational']} The answer is #### {demo['answer']}\n\n"
    prompt += f"Q:{sample['question']}\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"].replace(",", "")

def format_one_shot_aqua(demo, sample, tokenizer):
    prompt = f"Q:{demo['question']} \nA: {demo['rational']} The answer is: {demo['labels']}\n\n"
    prompt += f"Q:{sample['question']} {sample['choices']}\nA:"
    # tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"]

def format_8_shot_aqua(demo_list, sample, tokenizer):
    prompt = ""
    for demo in demo_list:
        prompt += f"Q:{demo['question']} \nA: {demo['rational']} The answer is: {demo['labels']}\n\n"
    prompt += f"Q:{sample['question']} {sample['choices']}\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"]

def format_one_shot_coin(demo, sample, tokenizer):
    prompt = f"Q:{demo['question']} \nA: {demo['rational']} The answer is: {demo['labels']}\n\n"
    prompt += f"Q:{sample['question']} \nA:"
    # tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"]

def format_8_shot_coin(demo_list, sample, tokenizer):
    prompt = ""
    for demo in demo_list:
        prompt += f"Q:{demo['question']} \nA: {demo['rational']} The answer is: {demo['labels']}\n\n"
    prompt += f"Q:{sample['question']} \nA:"
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"]





def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # load the dataset
    print("-----------loading dataset-----------")
    dataset_params = DATASET_PARAMS[args.dataset_name]
    if dataset_params["on_hugging_face"]:
        dataset = load_dataset(dataset_params["set_name"], dataset_params["config"])
    else:
        raise NotImplementedError("Only hugging face datasets are supported")
    test_dataset = dataset[dataset_params["evaluation_set"]]

    if "arc" in args.dataset_name or "csqa" in args.dataset_name:
        test_dataset = format_arc(test_dataset).select(range(700,1000))
    elif args.dataset_name == "gsm8k":
        test_dataset_full = format_gsm8k(test_dataset) #feature: question, rational, labels
        test_dataset = test_dataset_full.select(range(500,800)) # use 200 examples for evaluation
        #prompt_set = test_dataset_full["test"].select(range(-100,-1))
    elif args.dataset_name == "svamp":
        test_dataset = format_svamp(test_dataset) # already 300 rows
    elif args.dataset_name == "aqua":
        test_dataset = format_aqua(test_dataset)
    elif args.dataset_name == "coin_flip":
        test_dataset = format_coin(test_dataset).select(range(500,800))
    test_dataset = test_dataset.select(range(100,200))

    logging.info(f"Dataset: {args.dataset_name}")
    logging.info(f"Number of examples: {len(test_dataset)}")


    # load model and tokenizer
        

    bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    #bnb_4bit_use_double_quant=True,
    #bnb_4bit_quant_type="nf4",
    #bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("--------prepare model------------")

    if args.model_name is not None:
        if args.use_peft:
            logging.info(f"Use bnb config: {bnb_config}")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                use_cache=False,
                device_map=args.device,
                
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                use_cache=False,
                device_map=args.device,
                
            )
        if args.adapter_path is not None:
            logging.info(f"Adapter: {args.adapter_path}")
            model.load_adapter(args.adapter_path)
    else:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.adapter_path,
                use_cache=False,
                device_map=args.device,
                
            )
        except:
            raise ValueError("No model and adapter provided")
    
    #model=model.merge_and_unload()
        
    logging.info(f"Model: {args.model_name}")
    #logging.info(f"Adapter: {args.adapter_path}")
    logging.info(f"Final Model Type: {type(model)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              use_fast=False)
    



    # stopping criteria
    stop_words = ["\n\n"]
    if "Llama-3" in args.model_name:
        stop_word_idx = 1
    else:
        stop_word_idx = 2
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[stop_word_idx:] for stop_word in stop_words]
    #stop_words_ids.append(torch.tensor(696).unsqueeze(0))
    #stop_words_ids.append(torch.tensor(382).unsqueeze(0))


    stopping_criteria = StoppingCriteriaList([StoppingCriteria_token(stops=stop_words_ids)])




    # evaluate the model
    print("--------begin counting------------")
    
    
    if "arc" in args.dataset_name or "csqa" in args.dataset_name:
        logging.info(f"Evaluating on {args.dataset_name}")
        demo_list = [ARC["p1"], ARC["p2"], ARC["p3"], ARC["p4"], ARC["p5"], ARC["p6"], ARC["p7"], ARC["p8"]]
        prompt_formatter =  format_one_shot_arc
        prompt_formatter_8 = format_8_shot_arc
        evaluator = evaluate_arc
        
    elif args.dataset_name == "gsm8k" or args.dataset_name == "svamp":
        demo_list = [GSM8K["p1"],GSM8K["p2"],GSM8K["p3"],GSM8K["p4"],GSM8K["p5"],GSM8K["p6"],GSM8K["p7"],GSM8K["p8"]]
        prompt_formatter = format_one_shot_gsm8k
        prompt_formatter_8 = format_8_shot_gsm8k
        evaluator = evaluate_gsm8k

    elif args.dataset_name == "aqua":
        demo_list = [AQUA["p1"],AQUA["p2"],AQUA["p3"],AQUA["p4"],AQUA["p5"],AQUA["p6"],AQUA["p7"],AQUA["p8"]]
        prompt_formatter = format_one_shot_aqua
        prompt_formatter_8 = format_8_shot_aqua
        evaluator = evaluate_aqua
    elif args.dataset_name == "coin_flip":
        demo_list = [COIN["p1"], COIN["p2"], COIN["p3"], COIN["p4"],COIN["p5"], COIN["p6"], COIN["p7"], COIN["p8"]
        ]
        prompt_formatter = format_one_shot_coin
        prompt_formatter_8 = format_8_shot_coin
        evaluator = evaluate_coin

    

    res_list = []
    for i in tqdm(range(len(test_dataset))):
        res = []
        sample = test_dataset[i]
        
        if args.use_8_shot:
            prompt, input_ids, label = prompt_formatter_8(demo_list, sample, tokenizer)
            res.append(evaluator(input_ids, label, model, tokenizer, stopping_criteria))
        else:
            for demo in demo_list:
                prompt, input_ids, label = prompt_formatter(demo, sample, tokenizer)
                res.append(evaluator(input_ids, label, model, tokenizer, stopping_criteria))
        # count the number of True in res:
        print(f"num_pos_prompt for sample {i}: sum(res) = {sum(res)}")
        res_list.append(sum(res))
        
    test_dataset = test_dataset.add_column("num_pos_prompt", res_list) 
    test_dataset.save_to_disk(f"{args.output_name}")

if __name__ == "__main__":
    main()
