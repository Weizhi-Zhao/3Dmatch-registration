# 模式识别与机器学习课程设计

## 运行方法：
### 一. 在默认参数下直接运行

​		默认参数：method='ransac', dataset='indoor'

```shell
python main.py
```

### Indoor数据集
```shell
python main.py --method ransac --dataset indoor
python main.py --method icp_point2point --dataset indoor
python main.py --method icp_point2plane --dataset indoor
python main.py --method none --dataset indoor
```

### Plane数据集
```shell
python main.py --method ransac --dataset plane
......
```

## 数据集
### Indoor数据集
1. 导入方法
    	解压data.zip, 把indoor文件夹放在`3Dmatch-loader/data/indoor`目录下
### Plane数据集
1. 导入方法
    	解压plane_data.zip，把plane_data文件夹放在`3Dmatch-loader/data`目录下，并将文件夹重命名为plane

2. 说明

    ​	plane_data对应飞机的点云数据集，其中plane_all对应整个飞机的数据，分别存储为ply（可视化）和csv（记录坐标数据）。

    ​	plane_nose、plane_body、plane_tail分别对应机首机身机尾三部分，同样存储ply和csv。

    ​	`/plane_all`里存储24片飞机点云(包含每种飞机型号的csv文件和pcd文件，例如：`1.pcd`, `1.csv`），`/plane_all/data`是24片点云，`/plane_all/RT`存储变换矩阵（例如`plane1to2.txt`，plane1to2表示从1变换到2）。

    ​	`/nose`，`/body`，`/tail`分别存储机首机身机尾，每部分十片点云。



