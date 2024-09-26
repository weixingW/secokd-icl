#import packages
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from transformers import AutoTokenizer,  AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser
from peft import AutoPeftModelForCausalLM
#os.chdir("/workspace/lora_kd")
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
from huggingface_hub import login
logging.basicConfig(level=logging.INFO)



available_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {available_device}")

# Define and parse arguments.
@dataclass
class ScriptArguments:

    model_name: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    output_dir: Optional[str] = field(default="./results/lora_kd_alpha0.5", metadata={"help": "the output directory"})
    output_name: Optional[str] = field(default="results", metadata={"help": "the output name"})
    wandb_run_name: Optional[str] = field(default="lora_kd", metadata={"help": "the wandb run name"})
    adapter_path: Optional[str] = field(default=None, metadata={"help": "the adapter path"})
    dataset_name: Optional[Union[list,str]] = field(
        default="arc_challenge", metadata={"help": "the dataset name, optionally a list of datasets"}
    )
    few_shot_number: Optional[int] = field(default=2, metadata={"help": "the number of examples to sample from the dataset"})
    device: Optional[str] = field(default=available_device, metadata={"help": "the device to use"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "use peft model"})




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
        test_dataset = test_dataset_full.select(range(300,600)) # use 200 examples for evaluation
        #prompt_set = test_dataset_full["test"].select(range(-100,-1))
    elif args.dataset_name == "svamp":
        test_dataset = format_svamp(test_dataset) # already 300 rows
    elif args.dataset_name == "aqua":
        test_dataset = format_aqua(test_dataset)
    elif args.dataset_name == "coin_flip":
        test_dataset = format_coin(test_dataset)


    logging.info(f"Dataset: {args.dataset_name}")
    logging.info(f"Number of examples: {len(test_dataset)}")


    # load model and tokenizer
        

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("--------prepare model------------")

    if args.model_name is not None:
        if args.use_peft:
            logging.info(f"Use bnb config: {bnb_config}")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                use_cache=False,
                device_map=args.device
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                use_cache=False,
                device_map=args.device
            )
        if args.adapter_path is not None:
            logging.info(f"Adapter: {args.adapter_path}")
            model.load_adapter(args.adapter_path)
    else:
        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.adapter_path,
                use_cache=False,
                device_map=args.device
            )
        except:
            raise ValueError("No model and adapter provided")
    
    
        
    logging.info(f"Model: {args.model_name}")
    #logging.info(f"Adapter: {args.adapter_path}")
    logging.info(f"Final Model Type: {type(model)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    stop_words = ["\n\n"]
    if "Llama-3" in args.model_name:
        stop_word_idx = 1
    else:
        stop_word_idx = 2
    stop_word_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[stop_word_idx:] for stop_word in stop_words]
    #stop_word_ids.append(torch.tensor(696).unsqueeze(0))
    #stop_word_ids.append(torch.tensor(382).unsqueeze(0))
    print(stop_word_ids)    

    # evaluate the model
    print("--------begin evaluation------------")
    num_runs=3
    num_shot=args.few_shot_number

    results = np.zeros((num_runs,num_shot+1))
    
    if "arc" in args.dataset_name or "csqa" in args.dataset_name:
        logging.info(f"Evaluating on {args.dataset_name}")
        for i in tqdm(range(num_runs)):
            print(f"------------------Run {i}---------------------")
            for shot_number in range(num_shot+1):
                results[i,shot_number]=statis_acc_arc(test_dataset, model, tokenizer, shot_number,stop_word_ids)
    elif args.dataset_name == "gsm8k" or args.dataset_name == "svamp":
        logging.info(f"Evaluating on {args.dataset_name}")
        for i in tqdm(range(num_runs)):
            print(f"------------------Run {i}---------------------")
            for shot_number in range(num_shot+1):
                results[i,shot_number]=statis_acc_gsm8k(test_dataset, model, tokenizer, shot_number,stop_word_ids)
    elif args.dataset_name == "aqua":
        logging.info(f"Evaluating on {args.dataset_name}")
        for i in tqdm(range(num_runs)):
            print(f"------------------Run {i}---------------------")
            for shot_number in range(num_shot+1):
                results[i,shot_number]=statis_acc_aqua(test_dataset, model, tokenizer, shot_number,stop_word_ids)
    elif args.dataset_name == "coin_flip":
        logging.info(f"Evaluating on {args.dataset_name}")
        for i in tqdm(range(num_runs)):
            print(f"------------------Run {i}---------------------")
            for shot_number in range(num_shot+1):
                results[i,shot_number]=statis_acc_coin(test_dataset, model, tokenizer, shot_number,stop_word_ids)
    
    # save results as json:
    print("--------saving results------------")
    
    eval_res_path = os.path.join(args.output_dir,"evaluation")
    if not os.path.exists(eval_res_path):
        # If it does not exist, create the directory
        os.makedirs(eval_res_path)
    np.save(f"{eval_res_path}/{args.output_name}.npy", results)
    logging.info(f"Results saved to {eval_res_path}/results.npy")
if __name__ == "__main__":
    main()


