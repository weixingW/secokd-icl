import torch
import torch.nn as nn
from typing import Any, Generator, TypeVar, Union, Tuple
from transformers import models
import transformers as tr
import numpy as np
import pdb
import peft

Model = Union[tr.PreTrainedModel, "tl.HookedTransformer"]
Norm = Union[torch.nn.LayerNorm, models.llama.modeling_llama.LlamaRMSNorm, nn.Module]




def get_layers(model: Model,name) -> Any:
    if type(model) is torch.nn.parallel.distributed.DistributedDataParallel:
        model = model.module
    if type(model) is peft.peft_model.PeftModelForCausalLM:
        model = model.base_model.model
    # now model should be llamaforcausalmodel
    model_with_module = model if name == "lm_head" else model.base_model
    return getattr(model_with_module, name)


def _sqz(x):
    if isinstance(x, torch.Tensor):
        return x
    try:
        return x[0]
    except:
        return x

def make_lens_hooks(
    model: Model,
    decoder_layer_name: str='layers',
    reverse_emb_layer_name: str="lm_head",
    verbose: bool = True,
):
    """
    Add hooks to the model to record the logits of each layer.
    """
    
    model._lens_decoder = get_layers(model,reverse_emb_layer_name)
    clear_lens_hooks(model)
    clear_logits(model)

    
    """
    for attr in ["_layer_logits", "_layer_logits_handles"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})
    """

    if not hasattr(model, "_layer_logits_handles"):
        setattr(model, "_layer_logits_handles", {})
    if not hasattr(model, "_layer_logits"):
        setattr(model, "_layer_logits", [])
    
        
    
    layers = get_layers(model, decoder_layer_name)

    def logits_hook(module, input, output) -> None:
            """
            output: (`torch.FloatTensor`): output hidden_state of shape (batch, seq_len, embed_dim)
            """
            decoder_in = _sqz(output)
            decoder_out = model._lens_decoder(decoder_in)
            #print(decoder_out.detach().numpy().shape)
            
            model._layer_logits.append(decoder_out)
            #print(len(model._layer_logits))

    
    for idx, layer in enumerate(layers):
        if idx%2:
            name=f"layer_{idx}"
        
            handle = layer.register_forward_hook(logits_hook)

            model._layer_logits_handles[name]=handle



def clear_logits(model):
    if hasattr(model, "_layer_logits"):
        model._layer_logits = []
def clear_lens_hooks(model):
    if hasattr(model, "_layer_logits_handles"):
        for k, v in model._layer_logits_handles.items():
            v.remove()

        ks = list(model._layer_logits_handles.keys())
        for k in ks:
            del model._layer_logits_handles[k]

def collect_logits(model,batch_size,seq_len,role)->Tuple[np.ndarray, list]:
    #if type(model) is torch.nn.parallel.distributed.DistributedDataParallel:
    #    model=model.module

    layer_names=[]
    for k, v in model._layer_logits_handles.items():
        layer_names.append(k)
    if role=="student":
        layers_logits = torch.cat(model._layer_logits, dim=0)[-16:,...]
    elif role=="teacher": 
        layers_logits = torch.cat(model._layer_logits, dim=0)[:,seq_len*(batch_size-1):,...]
    else:
        raise ValueError("role must be student or teacher")
    clear_logits(model)
    return layers_logits, layer_names
