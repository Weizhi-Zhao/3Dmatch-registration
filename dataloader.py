import numpy as np
import torch

def collate_fn(data):
    """
    功能: 将数据列表整理成批处理所需的格式。
    输入参数:
        data: 包含单个样本的元组(src_pcd, tgt_pcd, rot, trans, correspondences)的列表。
    返回值: 
        dict_inputs: 包含批处理后的点云数据、旋转矩阵、平移向量和对应点的字典。
    """
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
    """
    功能: 创建数据加载器。
    输入参数:
        dataset: 数据集对象。
        num_workers: 加载数据时使用的进程数。
        shuffle: 是否在每个epoch开始时打乱数据。
    返回值: 
        DataLoader对象，用于迭代访问数据集。
    """
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
