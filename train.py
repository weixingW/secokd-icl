#import packages
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from huggingface_hub import login
from typing import Optional, List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments, DataCollatorWithPadding, HfArgumentParser
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModelForCausalLM
from datasets import load_dataset
from lorakd.trainer import KDTrainer
from lorakd.math_trainer import MATHTrainer
from data import *
from data.dataset_params import DATASET_PARAMS
from logit_lens import *
import numpy as np
import pdb
import math
import re
from trl import SFTTrainer



# Define and parse arguments.
@dataclass
class ScriptArguments:

    student_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})
    teacher_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})
    output_dir: Optional[str] = field(default="./results/lora_kd_sft", metadata={"help": "the output directory"})
    wandb_proj_name: Optional[str] = field(default=None, metadata={"help": "wandb project name"})
    report_to: Optional[str] = field(default=None, metadata={"help": "the report to"})
    wandb_run_name: Optional[str] = field(default="lora_kd", metadata={"help": "the wandb run name"})
    alpha: Optional[float] = field(default=1, metadata={"help": "the weight of ce loss from language modeling, value from 0 to 1"})
    tl_path: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf", metadata={"help": "the tuned lens path"})
    dataset_name: Optional[str] = field(
        default="arc_challenge", metadata={"help": "the dataset name"}
    )
    use_single: Optional[bool] = field(default=True, metadata={"help": "whether to use single prompt for the student."})
    use_ckp: Optional[str] = field(default=None, metadata={"help": "the adapter path"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "use peft model"})
    validation_set: Optional[str] = field(default="test", metadata={"help": "the split to validation"})
    use_random: Optional[bool] = field(default=False, metadata={"help": "whether to use random prompt rather than the 8 demos form GSM8k paper"})
    temp: Optional[float] = field(default=1, metadata={"help": "temperature for kd loss"})
    drop_column: Optional[bool] = field(default=True, metadata={"help": "weather to drop unused columns"})
    train_batch_size: Optional[int] = field(default=10, metadata={"help": "the training batch size"})
    eval_batch_size: Optional[int] = field(default=10, metadata={"help": "the evaluation batch size"})
    use_sft: Optional[bool] = field(default=False, metadata={"help": "whether to use SFT training"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    train_epoch: Optional[int] = field(default=1, metadata={"help": "Training epoch"})
    device_map: Optional[str] = field(default="auto", metadata={"help": "the device map"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=20, metadata={"help": "the number of logging steps"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the number of evaluation steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=3, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "Whether to activate fp16 mixed precision"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Whether to activate bf16 mixed precision"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    mixed_precision: Optional[str] = field(default="bf16", metadata={"help": "Mixed precision training"})
    target_modules: Optional[List[str]] = field(default=None, metadata={"help": "Target modules for LoRA adapters"})





def load_and_merge_adapter(model, adapter_path,lora_config):
    """
    customer function to load and merge a LORA adapter, which is not supported by the HF adapter loading function.
    Done by adding prefix 'base_model.model.' to the keys of the model state dict.
    """
    model.load_adapter(adapter_path)
    model_stats = model.state_dict()
    for key in list(model_stats.keys()):
        model_stats["base_model.model."+key] = model_stats.pop(key)
    model = PeftModelForCausalLM(model,lora_config)
    model.load_state_dict(model_stats)
    model = model.merge_and_unload()
    return model



class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.temperature = temperature





def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")



def compute_metric_with_tokenizer_arc(tokenizer=None):
    """
    Accuracy metric for ARC dataset
    """
    def compute_metric(eval_pred):
        sequences, labels = eval_pred
        prediction_list = []
        for i in range(sequences.shape[0]):
            prediction = sequences[i,:]
            prediction = prediction[prediction != -100]
            prediction = prediction[-15:]
            prediction = tokenizer.decode(prediction)
            prediction_list.append(prediction)
        extracted_answer = extract_answer_arc(prediction_list)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        true_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        assert len(extracted_answer) == len(true_labels)
        is_correct = []
        for res, answer in zip(extracted_answer, true_labels):
            if res==answer.lower():
                is_correct.append(1)
            else:
                is_correct.append(0)
        return {"accuracy": sum(is_correct)/len(is_correct)}
    return compute_metric

def compute_metric_with_tokenizer_coin(tokenizer=None):
    """
    Accuracy metric for coin dataset
    """
    def compute_metric(eval_pred):
        sequences, labels = eval_pred
        prediction_list = []
        for i in range(sequences.shape[0]):
            prediction = sequences[i,:]
            prediction = prediction[prediction != -100]
            prediction = prediction[-15:]
            prediction = tokenizer.decode(prediction)
            prediction_list.append(prediction)
        extracted_answer = extract_answer_coin(prediction_list)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        true_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        assert len(extracted_answer) == len(true_labels)
        is_correct = []
        for res, answer in zip(extracted_answer, true_labels):
            if res==answer.lower():
                is_correct.append(1)
            else:
                is_correct.append(0)
        return {"accuracy": sum(is_correct)/len(is_correct)}
    return compute_metric


def compute_metric_with_tokenizer_gsm8k(tokenizer=None):
    """
    Accuracy metric for GSM8K dataset
    """
    def compute_metric(eval_pred):
        #pdb.set_trace()
        extracted_answer = []
        sequences, labels = eval_pred
        # logits has the shape (batch_size, sequence_length, num_classes)
        for i in range(sequences.shape[0]):
            prediction = sequences[i,:]
            prediction = prediction[prediction != -100]
            prediction = prediction[-15:]
            prediction = tokenizer.decode(prediction)
            extracted_answer.append(extract_answer(prediction))
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        true_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        is_correct = []
        for res, answer in zip(extracted_answer, true_labels):
            answer = answer.replace(",", "")
            answer = answer.replace("$","").strip()
            if np.isnan(res):
                is_correct.append(0)
            elif math.isclose(res, float(answer), abs_tol=0.1, rel_tol=0.1):
                is_correct.append(1)
            else:
                is_correct.append(0)
        return {"accuracy": sum(is_correct)/len(is_correct)}
    return compute_metric

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.wandb_proj_name:
        os.environ["WANDB_PROJECT"] = args.wandb_proj_name  # name of W&B project

    # prepare model & tokenizer
    print("--------prepare model------------")


    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[ "q_proj","k_proj","out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    student_model_id = args.student_name
    teacher_model_id = args.teacher_name
    if args.use_peft:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        bnb_config = None

    if args.use_ckp:
        student_model = AutoModelForCausalLM.from_pretrained(
                student_model_id,
                use_cache=False,
                quantization_config=bnb_config,
                device_map="auto",
                
            )
        print(f"Student model: {args.student_name}")
        print(f"Student adapter: {args.use_ckp}")
        
        student_model  = load_and_merge_adapter(student_model,args.use_ckp,lora_config)


    else:
        student_model = AutoModelForCausalLM.from_pretrained(
            student_model_id,
            quantization_config=bnb_config,
            use_cache=False,
            device_map="auto",
            
        )

    if not args.use_sft:
        teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_id,
                quantization_config=bnb_config,
                use_cache=False,
                device_map="auto",
                
        )
        print(f"Teacher model: {args.teacher_name}")
        
        if args.use_ckp:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_id,
                use_cache=False,
                quantization_config=bnb_config,
                device_map="auto"
            )
            print(f"Adapter: {args.use_ckp}")
            teacher_model.load_adapter(args.use_ckp)
            
        else:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_id,
                quantization_config=bnb_config,
                use_cache=False,
                device_map="auto",
                
            )
            
            
        

    tokenizer = AutoTokenizer.from_pretrained(student_model_id)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    tokenizer.padding_side = "left"
    #student_model.config.pad_token_id = tokenizer.bos_token_id
    
    stop_words = ["\n\n"]
    if "Llama-3" in args.student_name:
        stop_word_idx = 1
    else:
        stop_word_idx = 2
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[stop_word_idx:] for stop_word in stop_words]
    if "Llama-3" in args.teacher_name and "Llama-3" not in args.student_name:
        for stop_word in stop_words:
            stop_words_ids.append(tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[1:])
    #stop_words_ids.append([696])
    #prepare dataset:
    print("--------prepare dataset------------")
    dataset_config=DATASET_PARAMS[args.dataset_name]
    dataset = load_dataset(dataset_config["set_name"], dataset_config["config"])
    sft_valid_set = dataset_config["validation_set"]
    if args.dataset_name == "arc_challenge" or args.dataset_name == "arc_easy" or args.dataset_name == "csqa" :
        dataset = format_arc(dataset)
        map_args = {"tokenizer":tokenizer, "use_single":args.use_single}
        tokenized_datasets = dataset.map(process_arc,fn_kwargs=map_args)
        tokenized_datasets = tokenized_datasets.remove_columns(["question","choices","answerKey"])
        if args.dataset_name == "csqa":
            tokenized_datasets = tokenized_datasets.remove_columns(["question_concept"])
        train_set = tokenized_datasets["train"].select(range(800))
        train_set.set_format(type=train_set.format["type"], columns=list(train_set.features.keys()))
        compute_metric = compute_metric_with_tokenizer_arc(tokenizer=tokenizer)
        valid_set = tokenized_datasets["validation"].select(range(200))
        valid_set.set_format(type=valid_set.format["type"], columns=list(valid_set.features.keys()))
        dataset_formatter = arc_formatter
    elif args.dataset_name == "coin_flip":
        dataset = format_coin(dataset)
        map_args = {"tokenizer":tokenizer, "use_single":args.use_single}
        tokenized_datasets = dataset.map(process_coin,fn_kwargs=map_args)
        tokenized_datasets = tokenized_datasets.remove_columns(["question"])
        train_set = tokenized_datasets["train"].select(range(800))
        train_set.set_format(type=train_set.format["type"], columns=list(train_set.features.keys()))
        compute_metric = compute_metric_with_tokenizer_coin(tokenizer=tokenizer)
        valid_set = tokenized_datasets["test"].select(range(200))
        valid_set.set_format(type=valid_set.format["type"], columns=list(valid_set.features.keys()))
        dataset_formatter = coin_formatter
    elif args.dataset_name == "gsm8k":
        dataset = format_gsm8k(dataset)
        prompt_set = dataset["test"].select(range(700,1100))
        #pdb.set_trace()
        map_args = {"tokenizer":tokenizer,
                "use_random":args.use_random,
                "prompt_set":prompt_set,
                "use_single":args.use_single
                }
        tokenized_datasets = dataset.map(process_gsm8k,fn_kwargs=map_args)
        tokenized_datasets = tokenized_datasets.remove_columns(["question","answer","rational"])
        # restore ignored columns: (see https://lewtun.github.io/blog/til/nlp/huggingface/transformers/2021/01/15/til-recovering-hidden-trainer-columns.html)
        train_set = tokenized_datasets["train"].select(range(800))
        train_set.set_format(type=train_set.format["type"], columns=list(train_set.features.keys()))
        compute_metric = compute_metric_with_tokenizer_gsm8k(tokenizer=tokenizer)
        valid_set = tokenized_datasets["test"].select(range(200))
        valid_set.set_format(type=valid_set.format["type"], columns=list(valid_set.features.keys()))
        dataset_formatter = gsm8k_formatter
    elif args.dataset_name == "svamp":
        dataset = format_svamp(dataset)
        map_args = {"tokenizer":tokenizer, "use_single":args.use_single}
        tokenized_datasets = dataset.map(process_svamp,fn_kwargs=map_args)
        tokenized_datasets = tokenized_datasets.remove_columns(["question"])
        train_set = tokenized_datasets["train"]
        train_set.set_format(type=train_set.format["type"], columns=list(train_set.features.keys()))
        compute_metric = compute_metric_with_tokenizer_gsm8k(tokenizer=tokenizer)
        valid_set = tokenized_datasets["test"].select(range(200))
        valid_set.set_format(type=valid_set.format["type"], columns=list(valid_set.features.keys()))
        dataset_formatter = svamp_formatter
    elif args.dataset_name == "aqua":
        dataset = format_aqua(dataset)
        map_args = {"tokenizer":tokenizer, "use_single":args.use_single}
        tokenized_datasets = dataset.map(process_aqua,fn_kwargs=map_args)
        tokenized_datasets = tokenized_datasets.remove_columns(["question", "options","rationale","correct","choices"])
        train_set = tokenized_datasets["train"].select(range(800))
        train_set.set_format(type=train_set.format["type"], columns=list(train_set.features.keys()))
        compute_metric = compute_metric_with_tokenizer_arc(tokenizer=tokenizer)
        valid_set = tokenized_datasets["validation"].select(range(200))
        valid_set.set_format(type=valid_set.format["type"], columns=list(valid_set.features.keys()))
        dataset_formatter = aqua_formatter
    print(dataset)
    print(tokenized_datasets)


    # define training args
    training_args = DistillationTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.train_epoch,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=1e-4,
        warmup_ratio=0.02,
        #warmup_steps=50,
        #bf16=False, # ensure proper upcasting for compute dtypes
        #tf32=True,
        #fp16_full_eval=True,
        remove_unused_columns=args.drop_column,
        fp16=True,
        #bf16=True,
        optim="paged_adamw_32bit", # from the QLoRA paper
        seed=33,
        #log_level="info",
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps", 
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss" if args.use_sft else "accuracy",
        report_to=args.report_to,
        run_name=args.wandb_run_name,
        # distilation parameters
        alpha=args.alpha,
        temperature=args.temp,
        
        )
    # define data_collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='longest')


    # prepare lora models
    student_model = get_peft_model(student_model, lora_config)
    print_trainable_parameters(student_model)
    #teacher_model = get_peft_model(teacher_model, lora_config)

    # prepare trainer
    if args.use_sft: 
        print("-------using SFT trainer--------")
        max_seq_length = 2048
        tokenizer.padding_side = "right"
        trainer = SFTTrainer(
            model=student_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset[sft_valid_set].select(range(50)),
            peft_config=lora_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            formatting_func=dataset_formatter,
            args=training_args,
        )
    else:
        if args.dataset_name == "gsm8k" or args.dataset_name == "svamp" or args.dataset_name == "coin_flip":
            print("-------using the math trainer--------")
            trainer = MATHTrainer(
                student_model,
                training_args,
                teacher_model=teacher_model,
                train_dataset=train_set,
                eval_dataset=valid_set,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metric,
                stop_word_ids=stop_words_ids
            )
        elif args.dataset_name == "arc_challenge" or args.dataset_name == "arc_easy" or args.dataset_name =="csqa" or args.dataset_name =="aqua":
            print("-------using the normal trainer--------")
            trainer = KDTrainer(
                student_model,
                training_args,
                teacher_model=teacher_model,
                train_dataset=train_set,
                eval_dataset=valid_set,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metric,
                stop_word_ids=stop_words_ids
            )
    
    #start training
    print("-------start training--------")
    trainer.train()

    print("-------finish and save model------------")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
        main()
