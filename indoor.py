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
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,infos,config,data_augmentation=True):
        super().__init__()
        self.infos = infos
        self.base_dir = config.root
        self.overlap_radius = config.overlap_radius
        self.config = config

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self,item): 
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
