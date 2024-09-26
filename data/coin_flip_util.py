import os
import pandas as pd
import numpy as np
import yaml
import re
import torch
import math
from random import shuffle
from tqdm import tqdm
from datasets import Value

from transformers import  StoppingCriteriaList
from .data_util import StoppingCriteria_token

work_dir = os.getcwd()
with open(os.path.join(work_dir,'data/coin_flip.yaml'), 'r') as file:
    COIN = yaml.safe_load(file)





def format_coin(dataset):
    """
    format questions so they can fit into existing prompt formats
    Args:
        dataset: Dataset object from HuggingFace
    """
    def format_question(sample):
        sample["question"]=sample["inputs"][3:] # Start after "Q: "
        return sample
    dataset = dataset.map(format_question)
    dataset = dataset.rename_column("targets","labels")
    dataset = dataset.remove_columns(["inputs", "targets_vec"])
    return dataset

def process_coin(example, tokenizer, use_single=True):
    
    """Construct the prompt from the example
    There are two options to contruct the student prompt:
    1. use one of the full prompt.
    2. use one random prompt.
    """
    demo_list = [COIN["p1"], COIN["p2"], COIN["p3"], COIN["p4"],COIN["p5"], COIN["p6"], COIN["p7"], COIN["p8"]]


    # teacher uses all 8 demos
    prompt_teacher=""
    for demo in demo_list:
        prompt_teacher+=f"Q:{demo['question']} \nA: {demo['rational']} The answer is {demo['labels']}\n\n"
    prompt_teacher += f"Q:{example['question']}\nA:"
    
    if use_single:
        chosen = demo_list[np.random.randint(0,8)]
        prompt_student = f"Q:{chosen['question']} \nA: {chosen['rational']} The answer is {chosen['labels']}\n\n"
        prompt_student += f"Q:{example['question']} \nA:"
    else:
        copy_demo_list = demo_list.copy()
        for i in range(np.random.randint(1,4)):
            chosen = copy_demo_list.pop(np.random.randint(0,len(copy_demo_list)))
            prompt_student = f"Q:{chosen['question']} \nA: {chosen['rational']} The answer is {chosen['labels']}\n\n"
        prompt_student += f"Q:{example['question']} \nA:"

    tokenized_inputs = tokenizer(prompt_student, truncation=True, max_length=2048)
    teacher_input = tokenizer(prompt_teacher, truncation=True,padding="max_length", max_length=1600)
    tokenized_inputs["teacher_input"] = teacher_input["input_ids"]
    tokenized_inputs["teacher_attention_mask"] = teacher_input["attention_mask"]
    labels = tokenizer(example["labels"].replace("(","").replace(")",""), truncation=True, padding="max_length",  max_length=6)
    tokenized_inputs["labels"] = labels["input_ids"]
    return tokenized_inputs

def coin_formatter(example):
    """
    Formation fucntion for SFT training
    """
    demo_list = [COIN["p1"], COIN["p2"], COIN["p3"], COIN["p4"],COIN["p5"], COIN["p6"], COIN["p7"], COIN["p8"]]
    chosen = demo_list[np.random.randint(0,8)]
    prompt = f"Q:{chosen['question']} \nA: {chosen['rational']} The answer is {chosen['labels']}\n\n"
    prompt += f"Q:{example['question']} \nA: The answer is {example['labels']}\n\n"
    return [prompt]

def format_few_shot_coin(sample, tokenizer, shot_number=0):
    """
    Construct a few-shot learning prompt from the dataset according to the shot_number
    Sample #shot_number of examples from the dataset and concatenate them into a single prompt
    return the tokenizered prompt-sample and the label
    """
    # sample #shot_number of examples from the demo list
    demo_list = [COIN["p1"], COIN["p2"], COIN["p3"], COIN["p4"],COIN["p5"], COIN["p6"], COIN["p7"], COIN["p8"]]
    shuffle(demo_list)
    prompt = "" 
    for i in range(shot_number):
        prompt += f"Q:{demo_list[i]['question']} \nA: {demo_list[i]['rational']} The answer is: {demo_list[i]['labels']}\n\n"
    prompt += f"Q:{sample['question']} \nA:"
    # tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"]

def extract_answer_coin(predictions):
    """
    input: prediction that contains answer: 'yes\n\n or no\n\n'
    output: a list of answers
    """
    pred_list = []
    # check type of predictions, if not list, convert to list:
    if not isinstance(predictions, list):
        predictions = [predictions]
    for prediction in predictions:
        # use re to extract :yes or no
        match = re.search(r' yes| no', prediction)
        if match:
            pred_list.append(match.group(0).lower().strip())
        else:
            pred_list.append("INVALID")
    return pred_list

def evaluate_coin(input_ids, label, model, tokenizer,stopping_criteria):
    """
    Evaluate the model on the prompt
    Args:
        prompt: string
        input_ids: tokenized prompt
        label: correct answer
        model: model
        tokenizer: tokenizer
    """
    input_ids = input_ids.to(model.device)
    # generate next token and decode:
    #pdb.set_trace()
    output = model.generate(**input_ids, do_sample=False,max_new_tokens=200, pad_token_id=tokenizer.bos_token_id, stopping_criteria=stopping_criteria)
    output = tokenizer.decode(output[-1][-15:])
    answer = extract_answer_coin(output)[0]
    #print(f"correct: {label}, got: {answer} from: {output}")
    return int(answer==label)

def statis_acc_coin(dataset, model, tokenizer, shot_number=2,stop_word_ids=None, sample_number=100):
    """
    Evaluate the model on the dataset
    Args:
        dataset: dataset
        model: model
        tokenizer: tokenizer
        shot_number: number of examples to sample from the dataset
    """
    
    stopping_criteria = StoppingCriteriaList([StoppingCriteria_token(stops=stop_word_ids)])
    correct = 0
    for i in range(sample_number):
        prompt, input_ids, label = format_few_shot_coin(dataset[i], tokenizer, shot_number)
        correct += evaluate_coin(input_ids, label, model, tokenizer,stopping_criteria=stopping_criteria)
    print(f"current shot: {shot_number}. current accuracy: {correct/sample_number}\n")
    return correct/50



