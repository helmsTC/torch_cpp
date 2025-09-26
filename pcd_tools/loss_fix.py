def _get_tgt_permutation_idx(self, indices, n_masks):
    # permute targets following indices
    batch_idx = torch.cat(
        [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
    )
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    
    # Handle empty case
    if batch_idx.numel() == 0:
        return []
    
    # From [B,id] to [id] of stacked masks
    cont_id = torch.cat([torch.arange(n) for n in n_masks if n > 0])
    
    # Safe max computation
    max_batch = batch_idx.max().item() if batch_idx.numel() > 0 else 0
    max_n = max(n_masks) if n_masks else 1
    
    b_id = torch.stack((batch_idx, cont_id), axis=1)
    map_m = torch.zeros((max_batch + 1, max_n))
    for i in range(len(b_id)):
        map_m[b_id[i, 0], b_id[i, 1]] = i
    stack_ids = [
        int(map_m[batch_idx[i], tgt_idx[i]]) for i in range(len(batch_idx))
    ]
    return stack_ids
