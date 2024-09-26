import numpy as np
import torch
from typing import List

def rank_logits_multi_token(new_logits:torch.Tensor, old_logits:torch.Tensor,topk:int=50) -> torch.Tensor:
    """
    Input: 
        new_logits & old_logits: two dimension tensor(seq_len,vocab)
    Output:
        two dimension tensor
    compare the new and old logits in the last dimension and return the top 50 largest absolute change in a mask
    """
    # Calculate the absolute change in logits
    change_in_logits = torch.abs(new_logits - old_logits)

    # get the indices of the top 50 changes for each row:
    top_50_indices = torch.topk(change_in_logits, topk, dim=-1).indices

    # Create a mask of zeros with the same shape as change_in_logits
    mask = torch.zeros_like(change_in_logits)

    # Set the top 50 positions in the mask to 1 for each row
    for i in range(mask.shape[0]):
        mask[i, top_50_indices[i]] = 1

    return mask

def rank_logits(new_logits:torch.Tensor, old_logits:torch.Tensor,topk:int=50) -> torch.Tensor:
    """
    Input: 
        new_logits & old_logits: one dimension tensor(vocab)
    Output:
        two dimension tensor
    compare the new and old logits in the last dimension and return the top 50 largest absolute change in a mask
    """
    # Calculate the absolute change in logits
    change_in_logits = torch.abs(new_logits - old_logits)

    # get the indices of the top 50 changes for each row:
    top_50_indices = torch.topk(change_in_logits, topk, dim=-1).indices

    # Create a mask of zeros with the same shape as change_in_logits
    mask = torch.zeros_like(change_in_logits)

    # Set the top 50 positions in the mask to 1 
    mask[top_50_indices] = 1

    return mask





