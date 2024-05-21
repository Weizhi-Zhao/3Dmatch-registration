import numpy as np
import torch

def collate_fn(data):
    assert len(data) == 1, "由于苏黎世联邦理工的人偷懒，batch_size只能等于1"
    src_pcd, tgt_pcd, rot, trans, correspondences = data[0]
    batched_points_list = [src_pcd, tgt_pcd]
    batched_lengths_list = [len(src_pcd), len(tgt_pcd)]
    
    batched_points = torch.cat(batched_points_list, axis=0).float()
    batched_lengths = torch.tensor(batched_lengths_list).int()

    dict_inputs = {
        'points': batched_points,
        'stack_lengths': batched_lengths,
        'rot': rot,
        'trans': trans,
        'correspondences': correspondences,
        'src_pcd_raw': src_pcd,
        'tgt_pcd_raw': tgt_pcd
    }

    return dict_inputs

def get_dataloader(dataset, num_workers=4, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        # https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/4
        collate_fn=collate_fn,
        drop_last=False
    )
    return dataloader
