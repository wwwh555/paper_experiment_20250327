# -*- coding: utf-8 -*-
# @File    : evaluate.py
# @Time    : 2025/3/28 18:17
# @Author  : wuhao
# @Description:
#   评估函数

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def dunn_index(data, labels):
    """
        快速计算 Dunn 指数
        :param data: 样本数据，形状为 (n_samples, n_features)
        :param labels: 聚类标签，形状为 (n_samples,)
        :return: Dunn 指数
        """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # 计算类内最大直径
    intra_dists = []
    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 1:
            dist_matrix = cdist(cluster_points, cluster_points)
            intra_dists.append(np.max(dist_matrix))
        else:
            intra_dists.append(0.0)  # 单点簇直径为0

    max_intra_dist = np.max(intra_dists)

    # 计算类间最小距离
    inter_dists = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = data[labels == unique_labels[i]]
            cluster_j = data[labels == unique_labels[j]]
            dist_matrix = cdist(cluster_i, cluster_j)
            inter_dists.append(np.min(dist_matrix))

    min_inter_dist = np.min(inter_dists) if inter_dists else 0.0

    # 计算 Dunn 指数
    return min_inter_dist / max_intra_dist if max_intra_dist != 0 else float('inf')


def accuracy_rate(true_labels, pred_labels):
    """
    计算聚类正确率（需真实标签）

    参数：
    true_labels : array-like, 形状 (n_samples,)
        真实标签（已知类别）
    pred_labels : array-like, 形状 (n_samples,)
        聚类预测标签

    返回：
    float : 聚类正确率（0~1之间）
    """
    # 确保输入为NumPy数组
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    # 检查输入长度一致
    assert len(true_labels) == len(pred_labels), "输入长度不一致"

    # 获取唯一标签
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)

    # 构建混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)

    # 使用匈牙利算法找到最佳标签匹配
    row_ind, col_ind = linear_sum_assignment(-cm)  # 取负号转为最大化问题

    # 计算正确分类的总数
    correct = cm[row_ind, col_ind].sum()

    # 计算正确率
    accuracy = correct / len(true_labels)

    return np.round(accuracy, 4)  # 保留4位小数