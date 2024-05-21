"""
Scripts for pairwise registration demo

Author: 
Last modified: 
"""
import os, torch, time, shutil, json,glob,sys,copy, argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch import optim, nn
import open3d as o3d

from indoor import IndoorDataset
from dataloader import get_dataloader
from utils import load_obj, setup_seed,natural_key, load_config
from benchmark_utils import to_tsfm, to_o3d_pcd, get_blue, get_yellow, to_tensor
from omegaconf import OmegaConf

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


def main(demo_loader):
    for inputs in demo_loader:
        pcd = inputs['points']
        len_src = inputs['stack_lengths'][0]
        rot, trans = inputs['rot'], inputs['trans']
        # correspondence = inputs['correspondences']
        src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]

        tsfm = to_tsfm(rot, trans)
        draw_registration_result(src_pcd, tgt_pcd, tsfm)


if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help= 'Path to the config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config = edict(config)   # 字典

    # create dataset and dataloader
    info_train = load_obj(config.train_info)
    train_set = IndoorDataset(info_train, config)

    train_loader = get_dataloader(
        dataset=train_set,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # do pose estimation
    main(train_loader)
