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
    """
    功能: 绘制配准结果，将源点云和目标点云在配准后的位置显示出来。
    输入参数:
        src_raw: 源点云数据。
        tgt_raw: 目标点云数据。
        tsfm: 源点云到目标点云的变换矩阵。
    返回值: 无。
    """
    ########################################
    # 1. input point cloud
    # 将原始点云数据转换为Open3D点云格式
    src_pcd_before = to_o3d_pcd(src_raw)
    # 使用变换矩阵对源点云进行变换
    src_pcd_before.transform(tsfm)
    # 将目标点云数据转换为Open3D点云格式
    tgt_pcd_before = to_o3d_pcd(tgt_raw)
    # 为源点云和目标点云分别设置颜色
    src_pcd_before.paint_uniform_color(get_yellow())
    tgt_pcd_before.paint_uniform_color(get_blue())
    # 估计源点云和目标点云的法向量
    src_pcd_before.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50)
    ) # 找每一个点的法向量
    tgt_pcd_before.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50)
    )

    # 创建一个可视化窗口并设置其属性
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='Input', width=960, height=540, left=100, top=100)
    # 向可视化窗口中添加源点云和目标点云
    vis1.add_geometry(src_pcd_before)
    vis1.add_geometry(tgt_pcd_before)

    # 更新可视化窗口以显示点云，直到窗口被关闭
    while True:
        vis1.update_geometry(src_pcd_before)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
    # 销毁可视化窗口
    vis1.destroy_window()

# def visualize_point_cloud(file):
#     pcd = o3d.io.read_point_cloud(file)
#     o3d.visualization.draw_geometries([pcd])

def main(demo_loader, method="none", dataset="none"):
    """
    功能: 根据指定的数据集和配准方法，对点云进行配准，并显示结果。
    输入参数:
        demo_loader: 数据加载器，包含点云数据。
        method: 使用的配准方法。
        dataset: 使用的数据集名称。
    返回值: 无。
    """
    # 如果数据集是室内数据集
    if dataset == 'indoor':
        # 遍历数据加载器中的每个样本
        for inputs in demo_loader:
            # 提取点云数据和相关信息
            pcd = inputs['points']
            len_src = inputs['stack_lengths'][0]
            rot, trans = inputs['rot'], inputs['trans']
            # correspondence = inputs['correspondences']

            # 分割源点云和目标点云
            src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]

            # 根据旋转和平移信息生成变换矩阵
            tsfm = to_tsfm(rot, trans)
            # 使用指定的配准方法对源点云和目标点云进行配准
            tsfm = REGISTRATIONS[method](src_pcd, tgt_pcd)
            # 绘制配准结果
            draw_registration_result(src_pcd, tgt_pcd, tsfm)


    # 如果数据集是平面数据集
    elif dataset == 'plane':
        # 遍历数据加载器中的每个样本
        for inputs in demo_loader:
            # 提取源点云和目标点云数据及其旋转和平移信息
            src_pcd = inputs['src']
            tgt_pcds = inputs['tgt']
            rots = inputs['rot']
            transes = inputs['trans']
            # correspondences = inputs['correspondences']

            # 对源点云进行预处理
            src_pcd.squeeze_(0)
            # 遍历每个目标点云
            for i in range(len(tgt_pcds)):
                tgt_pcd = tgt_pcds[i]
                rot = rots[i]
                trans = transes[i]
                # correspondence = correspondences[i]

                # 生成真实的变换矩阵
                tsfm_gt = to_tsfm(rot, trans)
                # 对目标点云进行预处理
                tgt_pcd.squeeze_(0)
                # 使用指定的配准方法对源点云和目标点云进行配准
                tsfm_test = REGISTRATIONS[method](src_pcd, tgt_pcd)
                # 比较算法估计的变换矩阵和真实的变换矩阵之间的差异
                rot_error, trans_error = compare_transformations(tsfm_test, tsfm_gt)
                # 打印旋转误差和平移误差
                print(f'Rotation error: {rot_error}, Translation error: {trans_error}')

                # 绘制配准结果
                draw_registration_result(src_pcd, tgt_pcd, tsfm_test)

# 将算法估计的旋转和平移同gt进行比较
def compare_transformations(tsfm_test, tsfm_gt):
    """
    功能: 比较算法估计的变换矩阵和真实的变换矩阵之间的差异。
    输入参数:
        tsfm_test: 算法估计的变换矩阵。
        tsfm_gt: 真实的变换矩阵（ground truth）。
    返回值:
        rot_error: 旋转误差。
        trans_error: 平移误差。
    """
    # 提取算法估计的旋转矩阵和平移向量
    est_rot = tsfm_test[:3, :3]
    est_trans = tsfm_test[:3, 3]
    # 提取真实的旋转矩阵和平移向量
    gt_rot = tsfm_gt[:3, :3]
    gt_trans = tsfm_gt[:3, 3]

    # 计算旋转矩阵之间的Frobenius范数，作为旋转误差
    rot_error = torch.norm(est_rot - gt_rot, p='fro')
    # 计算平移向量之间的L2范数，作为平移误差
    trans_error = torch.norm(est_trans - gt_trans, p=2)

    return rot_error, trans_error

'''
python main.py --method ransac
python main.py --method icp_point2point
python main.py --method icp_point2plane
python main.py --method none
'''

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ransac', help='ransac / icp_point2point / icp_point2plane / none')
    parser.add_argument('--dataset', type=str, default='plane', help='indoor / plane')
    args = parser.parse_args()

    # 加载配置文件
    config = load_config('./configs/train/'+args.dataset+'.yaml')
    config = edict(config)   # 字典

    # 根据命令行参数选择数据集并创建数据加载器
    if args.dataset == 'indoor':
        # 加载室内数据集的配置信息
        info_train = load_obj(config.train_info)    # train_info包含['src', 'tgt', 'rot', 'trans', 'overlap']
        train_set = IndoorDataset(info_train, config)
        # 创建数据加载器
        train_loader = get_dataloader(
            dataset=train_set,
            shuffle=False,
            num_workers=config.num_workers,
        )
    elif args.dataset == 'plane':
        # 加载平面数据集
        train_set = PlaneDataset(config)
        # 创建数据加载器
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers
        )
    else:
        # 如果数据集名称无效，抛出异常
        raise ValueError('Invalid dataset name')

    # 执行主函数，进行姿态估计
    main(train_loader, method=args.method, dataset=args.dataset)


