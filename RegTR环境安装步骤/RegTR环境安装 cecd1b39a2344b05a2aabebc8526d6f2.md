# RegTR环境安装

## RegTR网址

[https://github.com/yewzijian/RegTR](https://github.com/yewzijian/RegTR)

## RegTR/src/data_loaders/threedmatch.py的环境搭建过程

退出环境

```cpp
conda deactivate
```

创建激活环境

```cpp
conda create -n py38env python=3.8 //py3.9无法下载open3d？
// conda activate env_name
conda activate py38env
```

在conda的env中安装CUDA （cudatoolkit）

```cpp
conda install cudatoolkit=11.8

// 查看版本
conda list cudatoolkit
```

conda安装包

```cpp
conda install h5py=3.11.0

conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
```

失败deMan，但毕竟是3.8的失败的继续，所以坚持这么跑完也没有报错了，但没必要，转到正确的道路上吧。

```cpp
// 这部分的代码不用运行了
~~conda install -c open3d-admin open3d~~
// 然后一运行open3d这里就开始一直报：缺少各种包
// 包括但不限于：
~~conda install scikit-learn
conda install -c conda-forge addict
conda install pandas
conda install -c conda-forge plyfile
conda install -c conda-forge tqdm~~
```

## OMG好像Python3.9又可以了？发生了什么？

[Open3D – A  Modern Library for 3D Data Processing](https://www.open3d.org/)

本来是3.9无法下载open3d，突然又可以了

```cpp
conda install -c open3d-admin open3d
```

也不是很可以，conda勉强给我下了个0.15的版本，不是0.18的

```cpp
[Warning] Since Open3D 0.15, installing Open3D via conda is deprecated. Please re-install Open3D via: `pip install open3d -U`. 
// ！！早说啊installing Open3D via conda is deprecated. installing Open3D via conda is deprecated…………
conda remove -n myenv open3d=0.15.1
```

## 这下总可以了吧

还是在myenv中，但是pip安装^^

```cpp
pip install --upgrade pip
pip install open3d==0.18.0
```

验证Open3D是否安装在了正确的conda环境中：

```cpp
import open3d as o3d
print(o3d.__version__)
// 0.18.0
```

**It's over……^^**

## threedmatch.py修改调用module版

```python
"""Dataloader for 3DMatch dataset

Modified from Predator source code by Shengyu Huang:
  https://github.com/overlappredator/OverlapPredator/blob/main/datasets/indoor.py
    
Modified by AllenTU on 2024-5-11:
    combine the code of RegTR/src/data_loaders/threedmatch.py 
                        and /RegTR/src/data_loaders/se3_numpy.py
"""
_EPS = 1e-6
import logging
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# from se3_numpy import se3_init, se3_transform, se3_inv
# from pointcloud import compute_overlap

import math
from typing import List, Union

import torch.nn.functional as F
from torch import Tensor

import open3d as o3d
from typing import Union, Tuple

class ThreeDMatchDataset(Dataset):

    def __init__(self, cfg, phase, transforms=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            info_fname = f'datasets/3dmatch/{phase}_info.pkl'
            pairs_fname = f'{phase}_pairs-overlapmask.h5'
        else:
            info_fname = f'datasets/3dmatch/{phase}_{cfg.benchmark}_info.pkl'
            pairs_fname = f'{phase}_{cfg.benchmark}_pairs-overlapmask.h5'

        with open(info_fname, 'rb') as fid:
            self.infos = pickle.load(fid)

        self.base_dir = None
        if isinstance(cfg.root, str):
            if os.path.exists(f'{cfg.root}/train'):
                self.base_dir = cfg.root
        else:
            for r in cfg.root:
                if os.path.exists(f'{r}/train'):
                    self.base_dir = r
                break
        if self.base_dir is None:
            raise AssertionError(f'Dataset not found in {cfg.root}')
        else:
            self.logger.info(f'Loading data from {self.base_dir}')

        self.cfg = cfg

        if os.path.exists(os.path.join(self.base_dir, pairs_fname)):
            self.pairs_data = h5py.File(os.path.join(self.base_dir, pairs_fname), 'r')
        else:
            self.logger.warning(
                'Overlapping regions not precomputed. '
                'Run data_processing/compute_overlap_3dmatch.py to speed up data loading')
            self.pairs_data = None

        self.search_voxel_size = cfg.overlap_radius
        self.transforms = transforms
        self.phase = phase

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self, item):

        # get transformation and point cloud
        pose = se3_init(self.infos['rot'][item], self.infos['trans'][item])  # transforms src to tgt
        pose_inv = se3_inv(pose)
        src_path = self.infos['src'][item]
        tgt_path = self.infos['tgt'][item]
        src_xyz = torch.load(os.path.join(self.base_dir, src_path))
        tgt_xyz = torch.load(os.path.join(self.base_dir, tgt_path))
        overlap_p = self.infos['overlap'][item]

        # Get overlap region
        if self.pairs_data is None:
            src_overlap_mask, tgt_overlap_mask, src_tgt_corr = compute_overlap(
                se3_transform(pose, src_xyz),
                tgt_xyz,
                self.search_voxel_size,
            )
        else:
            src_overlap_mask = np.asarray(self.pairs_data[f'pair_{item:06d}/src_mask'])
            tgt_overlap_mask = np.asarray(self.pairs_data[f'pair_{item:06d}/tgt_mask'])
            src_tgt_corr = np.asarray(self.pairs_data[f'pair_{item:06d}/src_tgt_corr'])

        data_pair = {
            'src_xyz': torch.from_numpy(src_xyz).float(),
            'tgt_xyz': torch.from_numpy(tgt_xyz).float(),
            'src_overlap': torch.from_numpy(src_overlap_mask),
            'tgt_overlap': torch.from_numpy(tgt_overlap_mask),
            'correspondences': torch.from_numpy(src_tgt_corr),  # indices
            'pose': torch.from_numpy(pose).float(),
            'idx': item,
            'src_path': src_path,
            'tgt_path': tgt_path,
            'overlap_p': overlap_p,
        }

        if self.transforms is not None:
            self.transforms(data_pair)  # Apply data augmentation

        return data_pair
    
    # /src/data_loaders/se3_torch.py
    def se3_init(rot=None, trans=None):

        assert rot is not None or trans is not None

        if rot is not None and trans is not None:
            pose = torch.cat([rot, trans], dim=-1)
        elif rot is None:  # rotation not provided: will set to identity
            pose = F.pad(trans, (3, 0))
            pose[..., 0, 0] = pose[..., 1, 1] = pose[..., 2, 2] = 1.0
        elif trans is None:  # translation not provided: will set to zero
            pose = F.pad(rot, (0, 1))

        return pose

    def se3_cat(a, b):
        """Concatenates two SE3 transforms"""
        rot_a, trans_a = a[..., :3, :3], a[..., :3, 3:4]
        rot_b, trans_b = b[..., :3, :3], b[..., :3, 3:4]

        rot = rot_a @ rot_b
        trans = rot_a @ trans_b + trans_a
        dst = se3_init(rot, trans)
        return dst

    def se3_inv(pose):
        """Inverts the SE3 transform"""
        rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
        irot = rot.transpose(-1, -2)
        itrans = -irot @ trans
        return se3_init(irot, itrans)

    def se3_transform(pose, xyz):
        """Apply rigid transformation to points

        Args:
            pose: ([B,] 3, 4)
            xyz: ([B,] N, 3)

        Returns:

        """

        assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

        rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
        transformed = torch.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t

        return transformed

    def se3_transform_list(pose: Union[List[Tensor], Tensor], xyz: List[Tensor]):
        """Similar to se3_transform, but processes lists of tensors instead

        Args:
            pose: List of (3, 4)
            xyz: List of (N, 3)

        Returns:
            List of transformed xyz
        """

        B = len(xyz)
        assert all([xyz[b].shape[-1] == 3 and pose[b].shape[:-2] == xyz[b].shape[:-2] for b in range(B)])

        transformed_all = []
        for b in range(B):
            rot, trans = pose[b][..., :3, :3], pose[b][..., :3, 3:4]
            transformed = torch.einsum('...ij,...bj->...bi', rot, xyz[b]) + trans.transpose(-1, -2)  # Rx + t
            transformed_all.append(transformed)

        return transformed_all

    def se3_compare(a, b):
        combined = se3_cat(a, se3_inv(b))

        trace = combined[..., 0, 0] + combined[..., 1, 1] + combined[..., 2, 2]
        rot_err_deg = torch.acos(torch.clamp(0.5 * (trace - 1), -1., 1.)) \
                    * 180 / math.pi
        trans_err = torch.norm(combined[..., :, 3], dim=-1)

        err = {
            'rot_deg': rot_err_deg,
            'trans': trans_err
        }
        return err

    def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor = None):
        """Compute rigid transforms between two point sets

        Args:
            a (torch.Tensor): ([*,] N, 3) points
            b (torch.Tensor): ([*,] N, 3) points
            weights (torch.Tensor): ([*, ] N)

        Returns:
            Transform T ([*,] 3, 4) to get from a to b, i.e. T*a = b
        """

        assert a.shape == b.shape
        assert a.shape[-1] == 3

        if weights is not None:
            assert a.shape[:-1] == weights.shape
            assert weights.min() >= 0 and weights.max() <= 1

            weights_normalized = weights[..., None] / \
                                torch.clamp_min(torch.sum(weights, dim=-1, keepdim=True)[..., None], _EPS)
            centroid_a = torch.sum(a * weights_normalized, dim=-2)
            centroid_b = torch.sum(b * weights_normalized, dim=-2)
            a_centered = a - centroid_a[..., None, :]
            b_centered = b - centroid_b[..., None, :]
            cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)
        else:
            centroid_a = torch.mean(a, dim=-2)
            centroid_b = torch.mean(b, dim=-2)
            a_centered = a - centroid_a[..., None, :]
            b_centered = b - centroid_b[..., None, :]
            cov = a_centered.transpose(-2, -1) @ b_centered

        # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
        # and choose based on determinant to avoid flips
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
        rot_mat_pos = v @ u.transpose(-1, -2)
        v_neg = v.clone()
        v_neg[..., 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2)
        rot_mat = torch.where(torch.det(rot_mat_pos)[..., None, None] > 0, rot_mat_pos, rot_mat_neg)

        # Compute translation (uncenter centroid)
        translation = -rot_mat @ centroid_a[..., :, None] + centroid_b[..., :, None]

        transform = torch.cat((rot_mat, translation), dim=-1)
        return transform
    
    
    
    
    
    
    # /src/data_loaders/pointcloud.py        
    def compute_overlap(src: Union[np.ndarray, o3d.geometry.PointCloud],
                    tgt: Union[np.ndarray, o3d.geometry.PointCloud],
                    search_voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes region of overlap between two point clouds.

        Args:
            src: Source point cloud, either a numpy array of shape (N, 3) or
            Open3D PointCloud object
            tgt: Target point cloud similar to src.
            search_voxel_size: Search radius

        Returns:
            has_corr_src: Whether each source point is in the overlap region
            has_corr_tgt: Whether each target point is in the overlap region
            src_tgt_corr: Indices of source to target correspondences
        """

        if isinstance(src, np.ndarray):
            src_pcd = to_o3d_pcd(src)
            src_xyz = src
        else:
            src_pcd = src
            src_xyz = np.asarray(src.points)

        if isinstance(tgt, np.ndarray):
            tgt_pcd = to_o3d_pcd(tgt)
            tgt_xyz = tgt
        else:
            tgt_pcd = tgt
            tgt_xyz = tgt.points

        # Check which points in tgt has a correspondence (i.e. point nearby) in the src,
        # and then in the other direction. As long there's a point nearby, it's
        # considered to be in the overlap region. For correspondences, we require a stronger
        # condition of being mutual matches
        tgt_corr = np.full(tgt_xyz.shape[0], -1)
        pcd_tree = o3d.geometry.KDTreeFlann(src_pcd)
        for i, t in enumerate(tgt_xyz):
            num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(t, search_voxel_size)
            if num_knn > 0:
                tgt_corr[i] = knn_indices[0]
        src_corr = np.full(src_xyz.shape[0], -1)
        pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
        for i, s in enumerate(src_xyz):
            num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(s, search_voxel_size)
            if num_knn > 0:
                src_corr[i] = knn_indices[0]

        # Compute mutual correspondences
        src_corr_is_mutual = np.logical_and(tgt_corr[src_corr] == np.arange(len(src_corr)),
                                            src_corr > 0)
        src_tgt_corr = np.stack([np.nonzero(src_corr_is_mutual)[0],
                                src_corr[src_corr_is_mutual]])

        has_corr_src = src_corr >= 0
        has_corr_tgt = tgt_corr >= 0

        return has_corr_src, has_corr_tgt, src_tgt_corr
    
    def to_o3d_pcd(xyz, colors=None, normals=None):
        """
        Convert tensor/array to open3d PointCloud
        xyz:       [N, 3]
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)

        return pcd

```

跳转正确的道路

![Untitled](RegTR%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85%20cecd1b39a2344b05a2aabebc8526d6f2/Untitled.png)

名可夫斯基有个.so的大问题

解决这个大问题之后

PyTorch和PyTorch3D会打架——

·先下载Pytorch然后再3D，会：

![Untitled](RegTR%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85%20cecd1b39a2344b05a2aabebc8526d6f2/Untitled%201.png)

·先下载3D然后再PyTorch——其中，3D会携带PyTorch，我先删掉了携带的PyTorch，然后用conda再下的PyTorch：

![Untitled](RegTR%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85%20cecd1b39a2344b05a2aabebc8526d6f2/Untitled%202.png)

那不删东西了，就按顺序下——先下3D然后PyTorch

```cpp
conda install pytorch3d -c pytorch3d

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

```

> import torch
torch.cuda.is_available()
True
> 

其他：

pip和conda下的PyTorch3D不一样？

![Untitled](RegTR%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85%20cecd1b39a2344b05a2aabebc8526d6f2/Untitled%203.png)