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
work_dir = os.getcwd()
with open(os.path.join(work_dir,'data/gsm8k.yaml'), 'r') as file:
    GSM8K = yaml.safe_load(file)
with open(os.path.join(work_dir,'data/arc.yaml'), 'r') as file:
    ARC = yaml.safe_load(file)
from transformers import StoppingCriteria, StoppingCriteriaList





class StoppingCriteria_token(StoppingCriteria):

    def __init__(self, stops = []):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in input_ids:
            for stop in self.stops:
                if len(stop.shape)==0:
                    stop=stop.unsqueeze(0)
                if len(seq) >= len(stop) and torch.all((stop == seq[-len(stop):])).item():
                    #print(stop, seq[-len(stop):])
                    return True
        return False


def format_arc(dataset):
    """
    format questions so they can fit into existing prompt formats
    Args:
        dataset: Dataset object from HuggingFace
    """

    # remove the "id" column:
    dataset = dataset.remove_columns("id")

    def convert(item):
        to_letter = {"1":"(A)", "2":"(B)", "3":"(C)", "4":"(D)", "5":"(E)","A":"(A)","B":"(B)","C":"(C)","D":"(D)","E":"(E)"}
        return to_letter.get(item, item)
   
    def format_question(dataset):
        # remove periods
        text = dataset["choices"]["text"] # list of texts
        new_text = [str[:-1] if str[-1]=="." else str for str in text]
        dataset["choices"]["text"] = new_text
        #reformat
        option_format = "{} {}"
        dataset["choices"]["label"] = dataset["choices"]["label"] = [convert(i) for i in dataset["choices"]["label"]]
        options = [ option_format.format(dataset["choices"]["label"][i],dataset["choices"]["text"][i])for i in range(len(dataset["choices"]["label"]))]
        joined_options = ". ".join(options)
        dataset["choices"]=joined_options+"."

        dataset["labels"] = convert(dataset["answerKey"])#[convert(i) for i in dataset["answerKey"]]
        return dataset

    dataset = dataset.map(format_question)

    return dataset


def extract_answer_arc(predictions):
    """
    input: prediction that contains answer: 'xxAnswer:(a)\n\n'
    output: a 
    """
    pred_list = []
    # check type of predictions, if not list, convert to list:
    if not isinstance(predictions, list):
        predictions = [predictions]
    for prediction in predictions:
        # use re to extract :(a) or :a
        match = re.search(r"(\((\w)\))", prediction)
        if match:
            pred_list.append(match.group(2).lower())
        else:
            pred_list.append("INVALID")
    return pred_list
    

def process_arc(example, tokenizer=None, use_sys_prompt=False, use_random=False, use_single=True):
    demo_list = [ARC["p1"],ARC["p2"],ARC["p3"],ARC["p4"],ARC["p5"],ARC["p6"],ARC["p7"],ARC["p8"]]

    #teacher use all demo
    teacher_prompt = ""
    for demo in demo_list:
        teacher_prompt +=f"Question:{demo['question']}\nAnswer: {demo['rational']} The answer is: {demo['labels']}\n\n"
    teacher_prompt += f"Question:{example['question']} {example['choices']}\nAnswer:"
    
    #student use one random demo
    if use_random:
        chosen = demo_list[np.random.randint(0,8)]
        student_prompt = f"Question:{chosen['question']}\nAnswer: {chosen['rational']} The answer is: {chosen['labels']}\n\n"
        student_prompt += f"Question:{example['question']} {example['choices']}\nAnswer:"
    else:
        copy_demo_list = demo_list.copy()  
        for i in range(np.random.randint(1,4)):
            chosen = copy_demo_list.pop(np.random.randint(0,len(copy_demo_list)))
            student_prompt = f"Question:{chosen['question']}\nAnswer: {chosen['rational']} The answer is: {chosen['labels']}\n\n"
        student_prompt += f"Question:{example['question']} {example['choices']}\nAnswer:"


    tokenized_inputs = tokenizer(student_prompt, truncation=True, max_length=2048)
    teacher_input = tokenizer(teacher_prompt, truncation=True,padding="max_length", max_length=1700)
    tokenized_inputs["teacher_input"] = teacher_input["input_ids"]
    tokenized_inputs["teacher_attention_mask"] = teacher_input["attention_mask"]
    tokenized_inputs["label_position"] = torch.tensor(-5)
    labels = tokenizer(example["labels"].replace("(","").replace(")",""), truncation=True, padding=True, max_length=5)
    tokenized_inputs["labels"] = labels["input_ids"]
    return tokenized_inputs

def arc_formatter(example):
    demo_list = [ARC["p1"],ARC["p2"],ARC["p3"],ARC["p4"],ARC["p5"],ARC["p6"],ARC["p7"],ARC["p8"]]
    chosen = demo_list[np.random.randint(0,8)]
    prompt = f"Question:{chosen['question']}\nAnswer: {chosen['rational']} The answer is: {chosen['labels']}\n\n"
    prompt += f"Question:{example['question']} {example['choices']}\nAnswer: The answer is: {example['labels']}\n\n"
    return [prompt]

def format_few_shot_arc(sample, tokenizer, shot_number=0):
    """
    Construct a few-shot learning prompt from the dataset according to the shot_number
    Sample #shot_number of examples from the dataset and concatenate them into a single prompt
    return the tokenizered prompt-sample and the label
    """
    # sample #shot_number of examples from the demo list
    demo_list = [ARC["p1"],ARC["p2"],ARC["p3"],ARC["p4"],ARC["p5"],ARC["p6"],ARC["p7"],ARC["p8"]]
    shuffle(demo_list)
    prompt = "" 
    for i in range(shot_number):
        prompt += f"Question:{demo_list[i]['question']}\nAnswer: {demo_list[i]['rational']} The answer is: {demo_list[i]['labels']}\n\n"
    prompt += f"Question:{sample['question']} {sample['choices']}\nAnswer:"
    # tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"]

def evaluate_arc(input_ids, label, model, tokenizer,stopping_criteria):
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
    
    output = model.generate(**input_ids, do_sample=False,max_new_tokens=200, pad_token_id=tokenizer.bos_token_id, stopping_criteria=stopping_criteria)
    output = tokenizer.decode(output[-1][-15:])
    answer = extract_answer_arc(output)[0]
    label = label.replace("(","").replace(")","").strip().lower()
    #print(f"correct: {label}, got: {answer} from: {output}")
    return int(answer==label)

def statis_acc_arc(dataset, model, tokenizer, shot_number=2,stop_word_ids=None, sample_number=100):
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
        prompt, input_ids, label = format_few_shot_arc(dataset[i], tokenizer, shot_number)
        correct += evaluate_arc(input_ids, label, model, tokenizer,stopping_criteria=stopping_criteria)
    print(f"current shot: {shot_number}. current accuracy: {correct/sample_number}\n")
    return correct/sample_number


def format_gsm8k(dataset):
    def format_question(sample):
        sample["rational"]=sample["answer"].split('####')[-2].strip()
        sample["labels"] = sample["answer"].split('####')[-1].strip()
        return sample
    
    dataset = dataset.map(format_question)
    return dataset

def process_gsm8k(example, tokenizer, use_random=False, prompt_set = None, use_single=True):
    
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
    if use_random:
        len_prompt_set = len(prompt_set)
        random_sample = prompt_set[np.random.randint(0,len_prompt_set)]
        prompt_student = f"Q:{random_sample['question']}\nA: {chosen['rational']} The answer is #### {random_sample['labels']}\n\n"
        prompt_student += f"Q:{example['question']}\nA:"
    else:
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
    teacher_input = tokenizer(prompt_teacher, truncation=True,padding="max_length", max_length=1200)
    tokenized_inputs["teacher_input"] = teacher_input["input_ids"]
    tokenized_inputs["teacher_attention_mask"] = teacher_input["attention_mask"]
    tokenized_inputs["start_positions"] =torch.tensor(-ind)
    labels = tokenizer(example["labels"], truncation=True, padding="max_length",  max_length=6)
    tokenized_inputs["labels"] = labels["input_ids"]
    return tokenized_inputs

def gsm8k_formatter(example):
    demo_list = [GSM8K["p1"],GSM8K["p2"],GSM8K["p3"],GSM8K["p4"],GSM8K["p5"],GSM8K["p6"],GSM8K["p7"],GSM8K["p8"]]
    chosen = demo_list[np.random.randint(0,8)]
    prompt = f"Q:{chosen['question']}\nA: {chosen['rational']} The answer is #### {chosen['answer']}\n\n"
    prompt += f"Q:{example['question']}\nA: {example['rational']} The answer is #### {example['labels']}\n\n"
    return [prompt]


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE2 = re.compile(r"answer is (\-?[0-9\.\,]+)")
AN_RE3 = re.compile(r"= (\-?[0-9\.\,]+)")
INVALID_ANS = np.nan

def extract_answer(answer):
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

def statis_acc_gsm8k(dataset, model, tokenizer, shot_number=2,stop_word_ids=None, sample_number=100):
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
    for i in tqdm(range(sample_number)):
        prompt, input_ids, label = format_few_shot_gsm8k(dataset[i], tokenizer, shot_number)
        correct += evaluate_gsm8k(input_ids, label, model, tokenizer, stopping_criteria)
    print(f"current shot: {shot_number}. current accuracy: {correct/sample_number}\n")
    return correct/sample_number

def format_few_shot_gsm8k(sample, tokenizer, shot_number=0):
    """
    Construct a few-shot learning prompt from the dataset according to the shot_number
    Sample #shot_number of examples from the dataset and concatenate them into a single prompt
    return the tokenizered prompt-sample and the label
    """
    # sample #shot_number of examples from the dataset
    
    demo_list = [GSM8K["p1"],GSM8K["p2"],GSM8K["p3"],GSM8K["p4"],GSM8K["p5"],GSM8K["p6"],GSM8K["p7"],GSM8K["p8"]]
    # shuffle the demo list:
    shuffle(demo_list)
    prompt = "" 
    for i in range(shot_number):
        prompt += f"Q:{demo_list[i]['question']}"
        prompt += f"\nA: {demo_list[i]['rational']} The answer is #### {demo_list[i]['answer']}\n\n"
    prompt += f"Q:{sample['question']}\nA:"
    # tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt")
    return prompt, input_ids, sample["labels"].replace(",", "")

def evaluate_gsm8k(input_ids, label, model, tokenizer, stopping_criteria):
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
    outputs = model.generate(**input_ids, do_sample=False,max_new_tokens=600,pad_token_id=tokenizer.bos_token_id, stopping_criteria=stopping_criteria)
    answer = tokenizer.decode(outputs[-1][-15:])
    extracted_answer = extract_answer(answer)
    label = label.replace(",", "")
    label = label.replace("$","").strip()
    #print(f"correct: {label}, got: {extracted_answer} from {answer}")
    if extracted_answer == INVALID_ANS:
        return False
    elif math.isclose(extracted_answer, float(label), abs_tol=0.1, rel_tol=0.1):
        return True
    else:
        return False
    


