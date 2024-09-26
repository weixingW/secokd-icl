

# ARC_EASY
ARC_EASY_DATASET_PARAMS = {
    "set_name": "ai2_arc",
    "config": "ARC-Easy",
    "evaluation_set": "test",
    "validation_set": "validation",
    "on_hugging_face": True,
    "content_label_keys": (["question","choices"], "answerKey"),
}

#ARC_CHALLENGE
ARC_CHALLENGE_DATASET_PARAMS = {
    "set_name": "ai2_arc",
    "config": "ARC-Challenge",
    "evaluation_set": "test",
    "validation_set": "validation",
    "on_hugging_face": True,
    "content_label_keys": (["question","choices"], "answerKey"),
}

#CSQA
CSQA_PARAMS = {
    "set_name": "tau/commonsense_qa",
    "config": "default",
    "evaluation_set": "validation",
    "validation_set": "validation",
    "on_hugging_face": True,
    "content_label_keys": (["question","choices"], "answerKey"),
}

#GSM8K
GSM8K_DATASET_PARAMS = {
    "set_name": "gsm8k",
    "config": "main",
    "evaluation_set": "test",
    "validation_set": "test",
    "on_hugging_face": True,
    "content_label_keys": ("question", "answer"),
}

#SVAMP
SVAMP_PARAMS = {
    "set_name": "ChilleD/SVAMP",
    "config": "default",
    "evaluation_set": "test",
    "validation_set": "test",
    "on_hugging_face": True,

}

#AQUA
AQUA_PARAMS = {
    "set_name": "aqua_rat",
    "config": "raw",
    "evaluation_set": "test",
    "validation_set": "validation",
    "on_hugging_face": True,
}

# COIN_FLIP
COIN_FLIP_DATASET_PARAMS = {
    "set_name": "skrishna/coin_flip",
    "config": "default",
    "on_hugging_face": True,
    "evaluation_set": "test",
    "validation_set": "validation"
}


DATASET_PARAMS = {
    "arc_easy": ARC_EASY_DATASET_PARAMS,
    "arc_challenge": ARC_CHALLENGE_DATASET_PARAMS,
    "gsm8k": GSM8K_DATASET_PARAMS,
    "csqa": CSQA_PARAMS,
    "svamp":SVAMP_PARAMS,
    "aqua":AQUA_PARAMS,
    "coin_flip": COIN_FLIP_DATASET_PARAMS

}