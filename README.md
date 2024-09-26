# SeCoKD
This is the repository of the paper **SeCoKD: Aligning Large Language Models for In-Context Learning with
Fewer Shots**

SeCoKD provides a self-Knowledge Distillation training pipeline that aligns the student model with a heavily prompted variation, thereby increasing the utiliazaion of a single demonstration.
<p align="center">
  <img width="762" alt="Drawing (1)" src="https://github.com/weixingW/SeCoKD/assets/54362470/5f5a3e65-acba-4e6c-9feb-6f5f5a0b22e1">
</p>

## Highlights
- To the best of our knowledge, this work represents the first approach deliberately designed to reduce the number of demonstrations used for ICL by enhancing the model's ability to utilize a single demonstration.
- We design a KD training pipeline called SeCoKD and conduct comprehensive empirical experiments on various reasoning tasks in the ICL setting. In total, 6 datasets and 3 different models are used in this study.
- We investigate the robustness of SeCoKD in comparison to the SFT and show that our method not only provides more consistent improvements but also generalizes well to unseen tasks.

## Main Results
SeCoKD achieves significant improvement in zero-shot and one-shot learning
![Screen Shot 2024-06-13 at 1 26 47 PM](https://github.com/weixingW/SeCoKD/assets/54362470/09939ab7-b879-44bc-827b-77e42c4277ce)


## Environment
You should have Python3.10 and cuda driver supporting cuda118

To create the same environment Conda is highly recommended.

To use Conda simply run `conda env create -f environment.yml`.
This will create the right environment and run pip as subprocess to prepare all packages.

## Dataset
* [AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat?row=8)   `aqua`
* [ARC-C](https://huggingface.co/datasets/allenai/ai2_arc) `arc_challange`
* [CSQA](https://huggingface.co/datasets/skrishna/CSQA_preprocessed) `csqa`
* [COIN-FLIP](https://huggingface.co/datasets/skrishna/coin_flip?row=0) `coin_flip`
* [GSM8K](https://huggingface.co/datasets/openai/gsm8k) `gsm8k`
* [SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP) `svamp`

## Usage
First, decide which model and dataset to use. Currently, we only support huggingface models.

### Training
To train a new adapter using SeCoKD, you can run
```
python3 -m train  --student_name=$STUDENT \
                  --teacher_name=$TEACHER \
                  --use_peft=True  \
                  --output_dir=$OUTPUT_DIR\
                  --report_to=wandb \
                  --use_random=False \
                  --dataset_name=$DATASET_NAME \
                  --drop_column=False \
                  --train_batch_size=1 \
                  --train_epoch=10 \
                  --eval_batch_size=1 \      
```

To train a new adapter using SFT, you can run

```
python3 -m train  --model_name=$Model \
                  --use_peft=True \
                  --use_sft=True \
                  --output_dir=$OUTPUT_DIR\
                  --report_to=wandb \
                  --use_random=False \
                  --dataset_name=$DATASET_NAME \
                  --drop_column=True \
                  --train_batch_size=1 \
                  --train_epoch=10 \
                  --eval_batch_size=1 \  
```
We also support continued training, you can first train an adapter using one dataset and train another one on top of it. 

### Evaluation
You can use this script to evaluate the model
```
python3 -m evaluation  --model_name=$STUDENT \
                  --output_dir=$OUTPUT_DIR\
                  --output_name=$OUTPUT_NAME \
                  --use_peft=True \
                  --dataset_name=$DATASET_NAME \
                  --adapter_path=$PATH_TO_ADAPTER \
                  --few_shot_number=8
```

## License
This project is licensed under the MIT License.


## Roadmap
We will release pretrained adapters soon.


## Citation

