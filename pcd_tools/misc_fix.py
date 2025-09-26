from typing import List
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def pad_stack(tensor_list: List[Tensor]):
    """
    pad each tensor on the input to the max value in shape[1] and
    concatenate them in a single tensor.
    Input:
        list of tensors [Ni,Pi]
    Output:
        tensor [sum(Ni),max(Pi)]
    """
    if not tensor_list:  # Handle empty list
        return torch.tensor([])
    
    max_val = max([t.shape[1] for t in tensor_list])  # Fixed: *max -> max_val
    batched = torch.cat([F.pad(t, (0, max_val - t.shape[1])) for t in tensor_list])
    return batched

def sample_points(masks, masks_ids, n_pts, n_samples):
    """
    Select n_pts per mask to focus on instances
    plus random points up to n_samples
    """
    sampled = []
    for ids, mm in zip(masks_ids, masks):
        # Handle case when ids is empty (no instances in this sample)
        if len(ids) == 0:
            # Just sample random points when no instances exist
            r_idx = torch.randint(mm.shape[1], [n_samples])
            sampled.append(r_idx)
            continue
        
        # Sample points from instance masks
        instance_points = [
            id[torch.randperm(len(id))[:min(n_pts, len(id))]] 
            for id in ids if len(id) > 0  # Additional safety check
        ]
        
        # Handle case where all ids were empty after filtering
        if not instance_points:
            r_idx = torch.randint(mm.shape[1], [n_samples])
            sampled.append(r_idx)
            continue
        
        m_idx = torch.cat(instance_points)
        
        # Add random points to reach n_samples
        n_random = n_samples - m_idx.shape[0]
        if n_random > 0:
            r_idx = torch.randint(mm.shape[1], [n_random]).to(m_idx)
            idx = torch.cat((m_idx, r_idx))
        else:
            # If we already have enough instance points, truncate
            idx = m_idx[:n_samples]
        
        sampled.append(idx)
    
    return sampled
