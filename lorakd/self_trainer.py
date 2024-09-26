"""trainer class to perform lora kd for math tasks"""
import pdb
import gc
import sys
sys.path.append("..")
from logit_lens import *
from transformers import Trainer
from torch import nn
import numpy as np
from typing import Dict, Union, Any, Optional, List, Tuple
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available,is_peft_available
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


#from transformers.utils.trainer_pt_utils import smp_forward_backward
from transformers.trainer_pt_utils import nested_detach
if is_apex_available():
    from apex import amp
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
if is_peft_available():
    from peft import PeftModel
def _is_peft_model(model):
    return is_peft_available() and isinstance(model, PeftModel)




class SELFTrainer(Trainer):

    def __init__(self,*args, pad_token=1,teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        #NOTE: .to() is not applicable for bnb models, but it can be used for lora models, werid!
        #self._move_model_to_device(teacher_model,self.args.device)
        self.pad_token = pad_token
        

        


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        # register hooks
        make_lens_hooks(model)
        make_lens_hooks(self.teacher)

        model.train()
        inputs = self._prepare_inputs(inputs)
        """
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        """

        #with self.compute_loss_context_manager():
    
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach().to(self.args.device) / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        #pdb.set_trace()

        

        # prepare the student input: 
        input = inputs["input_ids"]

        # actual batch_size
        train_batch_size = input.shape[0]
        
        
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        

        
        student_out_idx = inputs.pop("student_token_len")
        student_labels = input['input_ids'][:,:student_out_idx]

        outputs_logits = model(input, return_dict=True)['logtis']
        student_ouput_logtis = outputs_logits[:,:student_out_idx,...] # (batch_size,first_seq_len, vocab)
        

        
        
        layer_names=[]
        for k, v in model._layer_logits_handles.items():
            layer_names.append(k)

        # start distill from the answer position：
        idx = inputs["start_positions"]
        


        #calculate the normal loss just like call model(**inputs) with labels
        if model_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = outputs_student['logits'][..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            student_loss = loss_fct(shift_logits, shift_labels)
        else:
            student_loss = torch.tensor(0.0).to(student_logits.device)
        student_loss=student_loss.to(model.device)
        
        # Soften probabilities and compute distillation loss
        """
        loss_function = nn.KLDivLoss(reduction="batchmean")
        logit_loss=torch.tensor(0.0).to(model.device)
        #pdb.set_trace()
        for i in range(train_batch_size):
            student_logits = torch.cat([logits[i,idx[i]:,...].unsqueeze(0) for logits in model._layer_logits],dim=0).to(model.device)
            #teacher_logits= torch.cat(self.teacher._layer_logits, dim=0)[:,-student_logits.shape[1]:,...].to(model.device)
            teacher_logits = torch.cat([logits[i,...].unsqueeze(0) for logits in self.teacher._layer_logits],dim=0).to(model.device)
            teacher_logits = teacher_logits[:,-student_logits.shape[1]:,...]
            # assert size
            assert student_logits.shape == teacher_logits.shape
            for j in range(len(layer_names)):
                logit_loss+=loss_function(
                    F.log_softmax(student_logits[j,...] / self.args.temperature, dim=-1),
                    F.softmax(teacher_logits[j,...] / self.args.temperature, dim=-1)) * (self.args.temperature ** 2)
        logit_loss = logit_loss/train_batch_size
        """
        loss_function = nn.MSELoss(reduction='mean')
        # might be better for regression tasks
        logit_loss = torch.tensor(0.).to(model.device)
        for i in range(train_batch_size):
            student_logits = torch.cat([logits[i,idx[i]:,...].unsqueeze(0) for logits in model._layer_logits],dim=0).to(model.device)
            # (layers,seq_answer,vocab)
            teacher_logits = torch.cat([logits[i,...].unsqueeze(0) for logits in self.teacher._layer_logits],dim=0).to(model.device)
            teacher_logits = teacher_logits[:,-student_logits.shape[1]:,...]
            # assert size
            assert student_logits.shape == teacher_logits.shape
            for j in range(len(layer_names)):
                # (seq_answer,vocab)
                logits_loss+=loss_function(
                    F.log_softmax(student_logits[j,...],dim=-1),
                    F.log_softmax(teacher_logits[j,...],dim=-1)
                )
        logit_loss = logit_loss/train_batch_size

        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * logit_loss

        # free mems
        del student_logits
        del teacher_logits
        gc.collect()

        return (loss, outputs_student) if return_outputs else loss



    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        #pdb.set_trace()
        make_lens_hooks(model)
        make_lens_hooks(self.teacher)

        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        # to use less space and make sure all logits have same shape we store the only last 20 tokens   
        logits = logits[:,-20:,:]
       
        return (loss, logits, labels)
    

    

