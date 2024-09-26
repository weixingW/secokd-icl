import os
import pandas as pd
import numpy as np
import yaml
import re
import pdb
import torch
import math
from random import shuffle
from tqdm import tqdm
from datasets import Value
work_dir = os.getcwd()
with open(os.path.join(work_dir,'data/gsm8k.yaml'), 'r') as file:
    GSM8K = yaml.safe_load(file)




def format_svamp(dataset):
    def format_question(sample):
        sample["question"]=sample["Body"]+sample["Question"]
        return sample
    dataset = dataset.map(format_question)
    dataset = dataset.rename_column("Answer","labels")
    dataset = dataset.remove_columns(["ID","Type", "Equation", "Body", "Question"])
    dataset = dataset.cast_column("labels", Value(dtype='string', id=None))
    return dataset

def process_svamp(example, tokenizer, use_single=True):
    
    """Construct the prompt from the example
    There are two options to contruct the student prompt:
    1. use one of the full prompt.
    2. use one random prompt.
    """
    prompt_list = [GSM8K["p1"],GSM8K["p2"],GSM8K["p3"],GSM8K["p4"],GSM8K["p5"],GSM8K["p6"],GSM8K["p7"],GSM8K["p8"]]
    


    # teacher uses all 8 demos
    prompt_teacher = GSM8K["full_prompt_cot"]
    prompt_teacher += f"Q:{example['question']}\nA:"
    
    # student prompt
    if use_single:
        chosen = prompt_list[np.random.randint(0,8)]
        prompt_student = f"Q:{chosen['question']}\nA: {chosen['rational']} The answer is #### {chosen['answer']}\n\n"
        prompt_student += f"Q:{example['question']}\nA:"
    else:
        copy_prompt_list = prompt_list.copy()
        for i in range(np.random.randint(1,4)):
            chosen = copy_prompt_list.pop(np.random.randint(0,len(copy_prompt_list)))
            prompt_student = f"Q:{chosen['question']}\nA: {chosen['rational']} The answer is #### {chosen['answer']}\n\n"
        prompt_student += f"Q:{example['question']}\nA:"
    # start position of the answer:
    ind = len(tokenizer(f"{example['labels']}\n\n", return_tensors="pt",truncation=True, max_length=32)["input_ids"].flatten())-2+1
    # -2 for starting token and " "
    # +1 for n->n+1

    tokenized_inputs = tokenizer(prompt_student, truncation=True, max_length=2048)
    teacher_input = tokenizer(prompt_teacher, truncation=True,padding="max_length", max_length=1400)
    tokenized_inputs["teacher_input"] = teacher_input["input_ids"]
    tokenized_inputs["teacher_attention_mask"] = teacher_input["attention_mask"]
    tokenized_inputs["start_positions"] =torch.tensor(-ind)
    labels = tokenizer(example["labels"], truncation=True, padding="max_length",  max_length=6)
    tokenized_inputs["labels"] = labels["input_ids"]
    return tokenized_inputs

def svamp_formatter(example):
    prompt_list = [GSM8K["p1"],GSM8K["p2"],GSM8K["p3"],GSM8K["p4"],GSM8K["p5"],GSM8K["p6"],GSM8K["p7"],GSM8K["p8"]]
    chosen = prompt_list[np.random.randint(0,8)]
    prompt = f"Q:{chosen['question']}\nA: {chosen['rational']} The answer is #### {chosen['answer']}\n\n"
    prompt += f"Q:{example['question']}\nA: The answer is #### {example['labels']}\n\n"
    return [prompt]

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE2 = re.compile(r"answer is (\-?[0-9\.\,]+)")
AN_RE3 = re.compile(r"= (\-?[0-9\.\,]+)")
INVALID_ANS = np.nan

def extract_answer_svamp(answer):
    """
    input: answer, a str containing ans
    output: extracted answer
    """
    
    completion = answer.replace("$","")
    completion = completion.replace(",", "")
    match = ANS_RE.search(completion)
    if not match:
        match = ANS_RE2.search(completion)
    if not match:
        match = AN_RE3.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.strip(".")
        try:
            ans = float(match_str)
            return ans
        except:
            return INVALID_ANS
                
    else:
        return INVALID_ANS


    
