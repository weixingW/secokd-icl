from typing import Any, Generator, TypeVar, Union, Tuple
from torch import nn
import transformers as tr
import torch
from transformers import models

Model = Union[tr.PreTrainedModel, "tl.HookedTransformer"]
Norm = Union[torch.nn.LayerNorm, models.llama.modeling_llama.LlamaRMSNorm, nn.Module]


def get_value_for_key(obj: Any, key: str) -> Any:
    """Get a value using `__getitem__` if `key` is numeric and `getattr` otherwise."""
    return obj[int(key)] if key.isdigit() else getattr(obj, key)


def get_key_path(model: th.nn.Module, key_path: str) -> Any:
    """Get a value by key path, e.g. `layers.0.attention.query.weight`."""
    for key in key_path.split("."):
        model = get_value_for_key(model, key)
    return model



def get_transformer_layers(model: Model) -> Tuple[str, torch.nn.ModuleList]:
    """Get the decoder layers from a model.

    Args:
        model: The model to search.

    Returns:
        A tuple containing the key path to the layer list and the list itself.

    Raises:
        ValueError: If no such list exists.
    """
    # TODO implement this so that we can do hooked transformer training.
    if not hasattr(model, "base_model"):
        raise ValueError("Model does not have a `base_model` attribute.")

    path_to_layers = ["base_model"]
    base_model = model.base_model
    if isinstance(base_model, models.opt.modeling_opt.OPTModel):
        path_to_layers += ["decoder", "layers"]
    elif isinstance(base_model, models.gpt_neox.modeling_gpt_neox.GPTNeoXModel):
        path_to_layers += ["layers"]
    elif isinstance(
        base_model,
        (
            models.bloom.modeling_bloom.BloomModel,
            models.gpt2.modeling_gpt2.GPT2Model,
            models.gpt_neo.modeling_gpt_neo.GPTNeoModel,
            models.gptj.modeling_gptj.GPTJModel,
        ),
    ):
        path_to_layers += ["h"]
    elif isinstance(base_model, models.llama.modeling_llama.LlamaModel):
        path_to_layers += ["layers"]
    else:
        raise NotImplementedError(f"Unknown model type {type(base_model)}")

    path_to_layers = ".".join(path_to_layers)
    return path_to_layers, get_key_path(model, path_to_layers)
