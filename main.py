"""
Scripts for pairwise registration demo

Author: 
Last modified: 
"""
import os, torch, time, shutil, json,glob,sys,copy, argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
import open3d as o3d

from indoor import IndoorDataset
from plane import PlaneDataset
from dataloader import get_dataloader
from utils import load_obj, setup_seed,natural_key, load_config
from benchmark_utils import to_tsfm, to_o3d_pcd, get_blue, get_yellow, to_tensor, get_correspondences
from omegaconf import OmegaConf
from registrations import REGISTRATIONS

import shutil

def draw_registration_result(src_raw, tgt_raw, tsfm):
    ########################################
    # 1. input point cloud
    src_pcd_before = to_o3d_pcd(src_raw)
    src_pcd_before.transform(tsfm)
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    src_pcd_before.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50)
    )  # 找每一个点的法向量
    tgt_pcd_before.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50)
    )

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=960, height=540, left=100, top=100)
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    while True:
        vis1.update_geometry(src_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()

    vis1.destroy_window()

# def visualize_point_cloud(file):
#     pcd = o3d.io.read_point_cloud(file)
#     o3d.visualization.draw_geometries([pcd])

def main(demo_loader, method="none", dataset="none"):
    if dataset == 'indoor':
        for inputs in demo_loader:
            pcd = inputs['points']
            len_src = inputs['stack_lengths'][0]
            rot, trans = inputs['rot'], inputs['trans']
            # correspondence = inputs['correspondences']
            src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]

            tsfm = to_tsfm(rot, trans)
            tsfm = REGISTRATIONS[method](src_pcd, tgt_pcd)
            draw_registration_result(src_pcd, tgt_pcd, tsfm)

    elif dataset == 'plane':
        for inputs in demo_loader:
            src_pcd = inputs['src']
            tgt_pcds = inputs['tgt']
            rots = inputs['rot']
            transes = inputs['trans']
            # correspondences = inputs['correspondences']

            src_pcd.squeeze_(0)
            for i in range(len(tgt_pcds)):
                tgt_pcd = tgt_pcds[i]
                rot = rots[i]
                trans = transes[i]
                # correspondence = correspondences[i]

                tsfm_gt = to_tsfm(rot, trans)
                tgt_pcd.squeeze_(0)
                tsfm_test = REGISTRATIONS[method](src_pcd, tgt_pcd)
                rot_error, trans_error = compare_transformations(tsfm_test, tsfm_gt)
                print(f'Rotation error: {rot_error}, Translation error: {trans_error}')

                draw_registration_result(src_pcd, tgt_pcd, tsfm_test)

# 将算法估计的旋转和平移同gt进行比较
def compare_transformations(tsfm_test, tsfm_gt):
    est_rot = tsfm_test[:3, :3]
    est_trans = tsfm_test[:3, 3]
    gt_rot = tsfm_gt[:3, :3]
    gt_trans = tsfm_gt[:3, 3]

    rot_error = torch.norm(est_rot - gt_rot, p='fro')
    trans_error = torch.norm(est_trans - gt_trans, p=2)

    return rot_error, trans_error

'''
python main.py --method ransac
python main.py --method icp_point2point
python main.py --method icp_point2plane
python main.py --method none
'''

if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ransac', help='ransac / icp_point2point / icp_point2plane / none')
    parser.add_argument('--dataset', type=str, default='plane', help='indoor / plane')

    args = parser.parse_args()
    config = load_config('./configs/train/'+args.dataset+'.yaml')
    config = edict(config)   # 字典

    # create dataset and dataloader
    if args.dataset == 'indoor':
        info_train = load_obj(config.train_info)    # train_info包含['src', 'tgt', 'rot', 'trans', 'overlap']
        train_set = IndoorDataset(info_train, config)
        train_loader = get_dataloader(
            dataset=train_set,
            shuffle=False,
            num_workers=config.num_workers,
        )
    elif args.dataset == 'plane':
        train_set = PlaneDataset(config)
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers
        )
    else:
        raise ValueError('Invalid dataset name')

    

    # do pose estimation
    main(train_loader, method=args.method, dataset=args.dataset)


