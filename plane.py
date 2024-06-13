import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences

class PlaneDataset(Dataset):
    def __init__(self, config):
        self.data_root = os.path.join(config.data_root, config.position)
        self.data_dir = os.path.join(self.data_root, 'data')
        self.rt_dir = os.path.join(self.data_root, 'RT')
        # self.files = sorted(os.listdir(self.data_dir))
        self.files = [f for f in sorted(os.listdir(self.data_dir)) if os.path.splitext(f)[1] == '.csv']
        self.overlap_radius = config.overlap_radius
        self.position = config.position

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        src_file = self.files[idx]
        src_pcd = pd.read_csv(os.path.join(self.data_dir, src_file)).values
        src_pcd = torch.tensor(src_pcd, dtype=torch.float64)

        tgt_pcds = []
        rots = []
        transes = []
        correspondences = []

        for i in range(len(self.files)):
            if i != idx:
                tgt_file = self.files[i]
                if self.position == 'plane_all':
                    rt_file = f'plane{os.path.splitext(src_file)[0]}to{os.path.splitext(tgt_file)[0]}.txt'
                else:
                    rt_file = f'{self.position}{os.path.splitext(src_file)[0]}to{os.path.splitext(tgt_file)[0]}.txt'
                    
                if os.path.exists(os.path.join(self.rt_dir, rt_file)):
                    tgt_pcd = pd.read_csv(os.path.join(self.data_dir, tgt_file)).values
                    rt = np.loadtxt(os.path.join(self.rt_dir, rt_file))

                    rot = rt[:3, :3]
                    trans = rt[:3, 3]

                    tgt_pcd = torch.tensor(tgt_pcd, dtype=torch.float64)
                    rot = torch.tensor(rot, dtype=torch.float32)
                    trans = torch.tensor(trans, dtype=torch.float32)

                    tgt_pcds.append(tgt_pcd)
                    rots.append(rot)
                    transes.append(trans)
                    correspondences.append(get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), rt, self.overlap_radius))

        # stack_lengths = [len(src_pcd), len(tgt_pcds[0])]

        return {
            'src': src_pcd,
            'tgt': tgt_pcds,
            'rot': rots,
            'trans': transes,
            'correspondences': correspondences
        }
    