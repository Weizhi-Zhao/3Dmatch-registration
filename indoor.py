"""
Author: HONGTOU TU
Last modified: 19.5.2024
"""

import os, torch
import numpy as np
from torch.utils.data import Dataset
from benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences


class IndoorDataset(Dataset):
    """
    功能: 为室内点云数据集提供一个数据加载接口。
    输入参数:
        infos: 包含数据集信息的字典。
        config: 包含数据集配置的对象。
        data_augmentation: 是否应用数据增强。
    方法:
        __len__: 返回数据集中样本的数量。
        __getitem__: 根据索引获取单个样本。
    备注: 
        Load subsampled coordinates, relative rotation and translation
        Output(torch.Tensor):
            src_pcd:        [N,3]
            tgt_pcd:        [M,3]
            rot:            [3,3]
            trans:          [3,1]
    """
    def __init__(self,infos,config,data_augmentation=True):
        """
        功能: 初始化IndoorDataset对象。
        输入参数:
            infos: 包含数据集信息的字典。
            config: 包含数据集配置的对象。
            data_augmentation: 是否应用数据增强。
        """
        super().__init__()
        self.infos = infos
        self.base_dir = config.root
        self.overlap_radius = config.overlap_radius
        self.config = config

    def __len__(self):
        """
        功能: 返回数据集中样本的数量。
        返回值: 
            数据集中样本的数量。
        """
        return len(self.infos['rot'])

    def __getitem__(self,item): 
        """
        功能: 根据索引获取单个样本。
        输入参数:
            item: 样本的索引。
        返回值: 
            包含源点云、目标点云、旋转矩阵、平移向量和对应点的元组。
        """
        # get transformation
        rot=self.infos['rot'][item]
        trans=self.infos['trans'][item]

        # get pointcloud
        src_path=os.path.join(self.base_dir,self.infos['src'][item])
        tgt_path=os.path.join(self.base_dir,self.infos['tgt'][item])
        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        if(trans.ndim==1):
            trans=trans[:,None]

        # get correspondence at fine level
        tsfm = to_tsfm(rot, trans)
        correspondences = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), tsfm,self.overlap_radius)
        rot = rot.astype(np.float32)
        rot = torch.from_numpy(rot)
        trans = trans.astype(np.float32)
        trans = torch.from_numpy(trans)

        return (
            torch.from_numpy(src_pcd),
            torch.from_numpy(tgt_pcd),
            rot,
            trans,
            correspondences,
        )
