# -*- coding: utf-8 -*-
# @File    : calculate.py
# @Time    : 2025/3/27 22:00
# @Author  : wuhao
# @Description:
#   计算函数

import numpy as np

'''
计算每个区间对应样本点特征的频率-即概率
'''


def calculated_frequency(feature, segment):
    # 样本数量
    num = len(feature)
    # 区间端点(有segment+1个)
    edge = np.linspace(0, 1, segment + 1)
    # 区间宽度
    width = edge[1] - edge[0]
    # 区间中心
    centers = edge[:-1] + width / 2
    # 统计区间特征数量
    counts = np.zeros(segment)

    for i in range(num):
        counts[int(feature[i] // width)] += 1  # 将特征除以width然后向下取整，即对应counts位置的

    return counts / num, centers, width
    # 返回概率，区间中点，以及区间宽度


'''
计算每个样本对应的概率值
'''
def calculate_sample_p(feature, probabilities, width):
    sample_num = len(feature)
    p = np.zeros(sample_num)
    for i in range(sample_num):
        p[i] = probabilities[int(feature[i] // width)]

    return p


'''
计算高斯核
'''


def gaussian_kernel(x1, x2, h):
    """
    计算两个向量之间的高斯核函数值。

    参数：
    x1 (numpy.ndarray): 输入向量1，形状为 (n_features,)
    x2 (numpy.ndarray): 输入向量2，形状为 (n_features,)
    h(float): 带宽参数

    返回：
    float: 高斯核函数值
    """
    # 计算欧氏距离的平方
    # sq_dist = np.linalg.norm(x1 - x2) ** 2

    sq_dist = np.sum((x1 - x2) ** 2)  # 避免精度损失？

    # 计算高斯核
    return np.exp(-sq_dist / (2 * (h ** 2)))


'''
 计算a和b向量之间的欧式距离
'''


def distance_of_points(a, b):
    return np.linalg.norm(a - b)


'''
计算point与data中所有点之间欧式距离<=h的点--邻域
'''


def group_point(point, data, h):
    m, n = data.shape

    res = []  # 响应结果集合
    for i in range(m):
        # if distance_of_points(point,data[i,:]) <= h:
        #     res.append(data[i,:].tolist())
        if np.sum((point - data[i, :]) ** 2) <= h ** 2:
            res.append(data[i, :].tolist())

    return np.array(res)


'''
计算重心
'''


def center_points(point, group, h):
    m, n = group.shape

    fenzi = np.zeros(len(point))
    fenmu = 0
    # 计算分母
    for i in range(m):
        now = group[i, :]
        w = gaussian_kernel(point, now, h)
        fenzi += w * now
        fenmu += w

    if fenmu < 1e-8:
        # 避免÷0
        return point  # 如果w太小，则重心就设置为point自己！
    return fenzi / fenmu


'''
均值漂移
'''


def mean_shift(data, h=0.2, max_iter=100, eps=1e-4):
    m, n = data.shape
    final_data = data.copy()  # 保存迭代过程中的点位置

    # 1.迭代
    for iter in range(max_iter):

        # 记录本次迭代的最大漂移量
        max_shift = 0.0

        # 1.1 遍历一遍每个样本点
        for i in range(m):
            point = final_data[i, :]

            # 1.2 计算邻域(遍历所有其他点，找到与当前点sample欧式距离)
            group = group_point(point, final_data, h)

            # 如果邻域没有点，直接跳过更新当前点
            if len(group) == 0:
                continue

            # 1.3 计算邻域内重心
            new_point = center_points(point, group, h)

            # 1.4 偏移向量的模
            shift = distance_of_points(point, new_point)

            # 1.5 更新本地迭代的最大漂移量
            if shift > max_shift:
                max_shift = shift

            # 1.6 将当前点移动到邻域重心，更新点
            final_data[i, :] = new_point

        # 1.7判断是否收敛
        print(f"Iteration {iter + 1}, Max Shift: {max_shift:.6f}")
        if max_shift < eps:
            break

    # 聚类合并：合并距离小于带宽的点
    labels = np.zeros(m, dtype=int)
    current_label = 0
    for i in range(m):
        if labels[i] == 0:  # 未被标记的点作为新类种子
            # 计算与当前点的距离
            distances = np.linalg.norm(final_data - final_data[i, :], axis=1)
            # 标记同一类
            labels[distances < h] = current_label
            current_label += 1

    return final_data, labels  # 返回均值漂移后所有点的新位置


'''
计算均值漂移的分组
'''


def groups(data, eps=0.1):
    """
    合并距离小于eps的点，分配聚类标签
    :param data: 收敛后的点坐标，形状为 (m, n)
    :param eps: 合并阈值
    :return: 聚类标签列表，形状为 (m,)
    """
    m = data.shape[0]
    labels = np.full(m, -1)  # 初始化标签为-1
    current_label = 0

    for i in range(m):
        if labels[i] == -1:  # 未标记的点
            # 找到所有与当前点距离 < eps的点
            distances = np.linalg.norm(data - data[i], axis=1)
            neighbors = np.where(distances < eps)[0]
            # 分配标签
            labels[neighbors] = current_label
            current_label += 1

    return labels.tolist()
