# -*- coding: utf-8 -*-
# @File    : radviz.py
# @Time    : 2025/3/27 16:08
# @Author  : wuhao
# @Description:
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from function.calculate import calculated_frequency


def radviz_coordinates(data, n_features):
    """
    计算Radviz坐标
    :param data: 归一化后的数据（二维数组，形状为 [n_samples, n_features]）
    :param features: 特征名称列表
    :return: 数据点在二维平面的坐标 (x, y)
    """
    # n_features = len(features)  # 特征数量
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)  # 锚点角度
    anchors = np.array([(np.cos(theta), np.sin(theta)) for theta in angles])  # 锚点坐标

    # 计算每个数据点的坐标
    sum_s = np.sum(data, axis=1)
    x = np.sum(data * anchors[:, 0], axis=1) / sum_s
    y = np.sum(data * anchors[:, 1], axis=1) / sum_s
    return x, y, anchors


def plot_radviz(x, y, anchors, feature_names, class_labels,
                segment_colors=None, class_colors=None, title="Radviz"):
    """
    绘制多颜色Radviz图
    - class_labels: 数据点类别标签（整数数组，如[0,1,2,...]）
    - anchor_colors: 锚点颜色列表（长度需与features一致）
    - class_colors: 类别颜色列表（长度需与类别数量一致）
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    n_feature = len(feature_names)

    # ---- 1. 绘制数据点（按类别着色） ----
    unique_classes = np.unique(class_labels)
    n_classes = len(unique_classes)

    # 自动生成颜色（若未指定）
    if class_colors is None:
        class_colors = plt.cm.tab10.colors[:n_classes]
    if segment_colors is None:
        segment_colors = plt.cm.tab20.colors[:n_feature]

    # 将类别标签映射为颜色数组
    color_map = {cls: class_colors[i] for i, cls in enumerate(unique_classes)}
    point_colors = [color_map[cls] for cls in class_labels]

    # 1.绘制样本散点图
    scatter = ax.scatter(x, y,
                         s=120,
                         c=point_colors,
                         alpha=0.7,
                         # edgecolors='#FFEDED',
                         # edgecolors='#413F3C',
                         edgecolors='#5C5A55',
                         linewidth=0.5)

    # ---- 2. 绘制锚点（每个特征不同颜色） ----
    for (ax_point, ay_point), feature in zip(anchors, feature_names):
        # 绘制锚点标记
        ax.scatter(ax_point, ay_point,
                   s=150,
                   # c='#FDFDFD',  # 使用统一颜色
                   c = '#000000',
                   marker='o',
                   # marker='s' # 使用正方形
                   edgecolor='#000000',
                   zorder=3)

        # 添加锚点标签
        ax.text(ax_point * 1.2,
                ay_point * 1.2,
                feature,
                ha='center',
                va='center',
                fontsize=14,
                color='#000000',  # 使用统一颜色
                weight='bold')

    # ---- 3. 添加圆形边框 ----
    # circle = plt.Circle((0, 0), 1.0, color='#ffc761', fill=False, linewidth=20, linestyle='--')
    # ax.add_artist(circle)
    angles = np.linspace(0, 2 * np.pi, n_feature, endpoint=False)

    # ---- 3. 绘制分段圆弧边框 ----
    arc_width = 25  # 线宽与原始设置一致
    arc_style = '-'  # 线型与原始设置一致
    gap_angle_deg = 2  # 白色间隙角度（单位：度）

    delta_theta_deg = 360 / n_feature  # 原始角度间隔

    for i in range(n_feature):
        # ---- 计算关键角度 ----
        # 当前锚点角度（弧度转角度）
        anchor_angle_deg = np.degrees(angles[i])

        # 颜色圆弧半跨度（扣除间隙后的半角）
        color_arc_half_span = (delta_theta_deg - gap_angle_deg) / 2

        # ---- 绘制颜色圆弧（以锚点为中心） ----
        color_start = anchor_angle_deg - color_arc_half_span
        color_end = anchor_angle_deg + color_arc_half_span
        arc = Arc((0, 0), 2, 2,
                  theta1=color_start,
                  theta2=color_end,
                  lw=arc_width,
                  linestyle=arc_style,
                  color=segment_colors[i],
                  zorder=2)
        ax.add_patch(arc)

        # ---- 绘制白色间隙（位于颜色圆弧右侧） ----
        white_start = color_end
        white_end = anchor_angle_deg + delta_theta_deg - color_arc_half_span  # 下一个锚点中心 - 半跨度
        white_arc = Arc((0, 0), 2, 2,
                        theta1=white_start,
                        theta2=white_end,
                        lw=arc_width,
                        linestyle=arc_style,
                        color='white',
                        zorder=3)
        ax.add_patch(white_arc)



    # ---- 4. 添加中心透明圆防止重叠 ----
    # center_circle = plt.Circle((0, 0), 0.98,
    #                            color='white',
    #                            fill=True,
    #                            zorder=2)
    # ax.add_artist(center_circle)

    # ---- 4. 样式优化 ----
    plt.axis('off')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    # ---- 5. 添加图例 ----
    # 数据类别图例
    # legend_elements = [
    #     plt.Line2D([0], [0], marker='o', color='w', label=f'Class {cls}',
    #                markerfacecolor=color_map[cls], markersize=8)
    #     for cls in unique_classes
    # ]
    # # 锚点特征图例
    # anchor_elements = [
    #     plt.Line2D([0], [0], marker='o', color='w', label=feature,
    #                markerfacecolor=anchor_colors[i], markersize=8)
    #     for i, feature in enumerate(features)
    # ]

    # ax.legend(handles=legend_elements + anchor_elements,
    #           loc='upper left', bbox_to_anchor=(1.05, 1),
    #           title="Legend", frameon=False)

    # plt.title(title, fontsize=14, pad=20)
    plt.show()



def plot_bar(data, feature_index, segment=50, color='#4ECDC4', alpha=0.7):

    #1.求直方图概率，中点，区间宽度
    probabilities, centers, width = calculated_frequency(data[:,feature_index], segment)

    # 5.绘图
    plt.figure(figsize=(8,5))
    plt.bar(centers,
            probabilities,
            width=width*0.9,  # 保留间隙
            edgecolor='white',
            linewidth=0.5,
            color=color,
            alpha=alpha)

    # 6.标注特征索引和参数
    plt.title(f'Feature {feature_index + 1} Probability Distribution (r={segment})',
              fontsize=14, pad=15)
    plt.xlabel('Normalized Value', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 7.显示前5个非零概率区间的数值标注
    # for i, (center, prob) in enumerate(zip(bin_centers, probabilities)):
    #     if prob > 0 and i < 5:  # 仅标注前5个非零值
    #         plt.text(center, prob + 0.002, f'{prob:.2f}',
    #                  ha='center', va='bottom', fontsize=8)

    # plt.xlim(np.linspace(0, 1, 11))
    plt.xticks(np.linspace(0, 1, 11))
    plt.ylim(0, probabilities.max()*1.2)
    plt.tight_layout()
    plt.show()


'''
画均值漂移(对原数据点)特征分段后的直方图
'''
def plot_group_bar(feature, feature_index, group, segment=50, alpha=0.7):

    colors = ['#FF6B6B', '#4ECDC4', '#ffc761', '#0080ff', '#45B7D1',  '#96CEB4', '#FF9999', '#66B2FF', ]
    # colors = plt.cm.get_cmap('viridis', 10)  # 'viridis' 是一种 colormap，7 表示采样 7 个颜色

    #1.取不同的分组
    # 获取唯一分组标签
    unique_groups = np.unique(group)
    group_num = len(unique_groups)
    # 存储分组结果
    grouped = {}
    for g in unique_groups:
        mask = (group == g)
        grouped[g] = feature[mask]

    # 2.求直方图概率，中点，区间宽度
    probabilities, centers, width = calculated_frequency(feature, segment)

    # 5.绘图
    plt.figure(feature_index)
    plt.figure(figsize=(8, 5))

    for i in range(group_num):
        # 遍历每个分组，然后画图

        now = grouped[unique_groups[i]]

        p = np.zeros(len(centers))

        for j in range(len(now)):
            # 遍历当前分组，计算对应p
            loc = int(now[j] // width)  # 这个点所在的区间索引
            p[loc] = probabilities[loc]

        plt.bar(centers,
                p,
                width=width*0.9,  # 保留间隙
                edgecolor='white',
                linewidth=0.5,
                color=colors[i],
                alpha=alpha)


    # 7.显示前5个非零概率区间的数值标注
    # for i, (center, prob) in enumerate(zip(bin_centers, probabilities)):
    #     if prob > 0 and i < 5:  # 仅标注前5个非零值
    #         plt.text(center, prob + 0.002, f'{prob:.2f}',
    #                  ha='center', va='bottom', fontsize=8)

    # 6.标注特征索引和参数
    plt.title(f'Feature {feature_index + 1} Probability Distribution (r={segment})',
              fontsize=14, pad=15)
    plt.xlabel('Normalized Value', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # plt.xlim(np.linspace(0, 1, 11))
    plt.xticks(np.linspace(0, 1, 11))
    plt.ylim(0, probabilities.max()*1.2)
    plt.tight_layout()
    plt.show()


'''
画均值漂移(对50个直方图)后的分段直方图
'''
def plot_group_bar2(feature_index, probabilities, centers, width, group, segment=50, alpha=0.7):

    colors = ['#FF6B6B', '#4ECDC4', '#ffc761', '#0080ff',
              '#45B7D1',  '#96CEB4', '#FF9999', '#66B2FF',
              '#00c6ba', '#0095fa', '#facdba', '#cf9eaa',
              '#835500', '#424751', '#797e89']
    # colors = plt.cm.get_cmap('viridis', 10)  # 'viridis' 是一种 colormap，7 表示采样 7 个颜色

    #1.取不同的分组
    # 获取唯一分组标签
    unique_groups = np.unique(group)
    group_num = len(unique_groups)
    # 存储分组结果
    grouped = {}
    for g in unique_groups:
        mask = (group == g)
        grouped[g] = centers[mask]

    # 5.绘图
    plt.figure(feature_index)
    plt.figure(figsize=(8, 5))

    for i in range(group_num):
        # 遍历每个分组，然后画图

        # 当前分组
        now = grouped[unique_groups[i]]

        p = np.zeros(len(centers))

        for j in range(len(now)):
            # 遍历当前分组，计算对应p
            loc = int(now[j] // width)  # 这个点所在的区间索引
            p[loc] = probabilities[loc]

        plt.bar(centers,
                p,
                width=width*0.9,  # 保留间隙
                edgecolor='white',
                linewidth=0.5,
                color=colors[i],
                alpha=alpha)


    # 7.显示前5个非零概率区间的数值标注
    # for i, (center, prob) in enumerate(zip(bin_centers, probabilities)):
    #     if prob > 0 and i < 5:  # 仅标注前5个非零值
    #         plt.text(center, prob + 0.002, f'{prob:.2f}',
    #                  ha='center', va='bottom', fontsize=8)

    # 6.标注特征索引和参数
    plt.title(f'Feature {feature_index + 1} Probability Distribution (r={segment})',
              fontsize=14, pad=15)
    plt.xlabel('Normalized Value', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # plt.xlim(np.linspace(0, 1, 11))
    plt.xticks(np.linspace(0, 1, 11))
    plt.ylim(0, probabilities.max()*1.2)
    plt.tight_layout()
    plt.show()


'''
画均值漂移(对50个直方图)后的分段直方图
'''
def plot_group_bar3(feature_index, probabilities, centers, width, group, segment=50, alpha=0.7):

    colors = ['#FF6B6B', '#4ECDC4', '#ffc761', '#0080ff',
              '#45B7D1',  '#96CEB4', '#FF9999', '#66B2FF',
              '#00c6ba', '#0095fa', '#facdba', '#cf9eaa',
              '#835500', '#424751', '#797e89']
    # colors = plt.cm.get_cmap('viridis', 10)  # 'viridis' 是一种 colormap，7 表示采样 7 个颜色

    #1.取不同的分组
    # 获取唯一分组标签
    unique_groups = np.unique(group)
    group_num = len(unique_groups)
    # 存储分组结果
    grouped = {}
    for g in unique_groups:
        mask = (group == g)
        grouped[g] = centers[mask]

    # 5.绘图
    plt.figure(feature_index)
    plt.figure(figsize=(8, 5))

    for i in range(group_num):
        # 遍历每个分组，然后画图

        # 当前分组
        now_group = unique_groups[i]
        # 获取当前分组的所有柱形中点
        now_centers = grouped[now_group]

        # 获取对应当前柱形对应概率p值
        index = group == now_group

        # p = np.zeros(len(now_centers))
        p = probabilities[index]

        plt.bar(now_centers,
                p,
                width=width*0.9,  # 保留间隙
                edgecolor='white',
                linewidth=0.5,
                color=colors[i],
                alpha=alpha)


    # 7.显示前5个非零概率区间的数值标注
    # for i, (center, prob) in enumerate(zip(bin_centers, probabilities)):
    #     if prob > 0 and i < 5:  # 仅标注前5个非零值
    #         plt.text(center, prob + 0.002, f'{prob:.2f}',
    #                  ha='center', va='bottom', fontsize=8)

    # 6.标注特征索引和参数
    plt.title(f'Feature {feature_index + 1} Probability Distribution (r={segment})',
              fontsize=14, pad=15)
    plt.xlabel('Normalized Value', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # plt.xlim(np.linspace(0, 1, 11))
    plt.xticks(np.linspace(0, 1, 11))
    plt.ylim(0, probabilities.max()*1.2)
    plt.tight_layout()
    plt.show()