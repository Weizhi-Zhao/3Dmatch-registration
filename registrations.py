import open3d as o3d
from typing import Optional
import numpy as np
from benchmark_utils import to_o3d_pcd


def ransac_registration(src_pcd, tgt_pcd, voxel_size=0.05):
    """
    功能: 使用RANSAC算法和FPFH特征进行点云的全局配准。
    输入参数:
        src_pcd: 源点云数据。
        tgt_pcd: 目标点云数据。
        voxel_size: 体素大小，用于估计法线和计算FPFH特征。
    返回值: 最佳变换矩阵，将源点云对齐到目标点云。
    """
    # 将输入的点云数据转换为Open3D点云格式
    src_pcd = to_o3d_pcd(src_pcd)
    tgt_pcd = to_o3d_pcd(tgt_pcd)
    
    # 估计源和目标点云的法线
    src_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # 计算源和目标点云的FPFH特征
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
    # 使用RANSAC算法和FPFH特征进行全局配准
    distance_threshold = voxel_size * 1.5
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=src_pcd, 
        target=tgt_pcd, 
        source_feature=src_fpfh, 
        target_feature=tgt_fpfh, 
        mutual_filter=True,   # 改
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return ransac_result.transformation

def icp_registraion_point2point(source, target, threshold=0.02, trans_init: Optional[np.ndarray]=None):
    """
    功能: 使用点对点ICP算法进行点云的精细配准。
    输入参数:
        source: 源点云数据。
        target: 目标点云数据。
        threshold: 最大对应点对距离阈值，用于配准。
        trans_init: 初始变换矩阵，默认为单位矩阵。
    返回值: 最佳变换矩阵，将源点云对齐到目标点云。
    """
    # 将输入的点云数据转换为Open3D点云格式
    source = to_o3d_pcd(source)
    target = to_o3d_pcd(target)
    
    # 如果未提供初始变换矩阵，则使用单位矩阵
    if trans_init is None:
        trans_init = np.eye(4)
    
    # 使用点对点ICP算法进行精细配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return reg_p2p.transformation


def icp_registraion_point2plane(source, target, threshold=0.02, trans_init: Optional[np.ndarray]=None):
    """
    功能: 使用点到平面ICP算法进行点云的精细配准。
    输入参数:
        source: 源点云数据。
        target: 目标点云数据，需要预先估计法线。
        threshold: 最大对应点对距离阈值，用于配准。
        trans_init: 初始变换矩阵，默认为单位矩阵。
    返回值: 最佳变换矩阵，将源点云对齐到目标点云。
    """
    # 将输入的点云数据转换为Open3D点云格式
    source = to_o3d_pcd(source)
    target = to_o3d_pcd(target)

    # 为目标点云估计法线
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # 如果未提供初始变换矩阵，则使用单位矩阵
    if trans_init is None:
        # trans_init = np.asarray(
        #     [
        #         [0.862, 0.011, -0.507, 0.5],
        #         [-0.139, 0.967, -0.215, 0.7],
        #         [0.487, 0.255, 0.835, -1.4],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ]
        # )
        trans_init = np.eye(4)
    # 使用点到平面ICP算法进行精细配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return reg_p2p.transformation


REGISTRATIONS = {
    "ransac": ransac_registration,
    "icp_point2point": icp_registraion_point2point,
    "icp_point2plane": icp_registraion_point2plane,
    "none": lambda s, t: np.eye(4), # 不进行配准，直接返回单位矩阵
}