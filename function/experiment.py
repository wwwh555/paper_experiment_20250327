# -*- coding: utf-8 -*-
# @File    : experiment.py
# @Time    : 2025/3/28 23:58
# @Author  : wuhao
# @Description:
#   实验函数

import numpy as np
import pandas as pd

from function.cyclic_permutation import generate_cyclic_permutations, permutation_to_feature_name
from function.evaluate import dunn_index, accuracy_rate
from function.k_means_cluster import k_means_cluster
from function.generate_color import segment_color_generate, class_color_generate
from function.radviz import radviz_coordinates, plot_radviz


def perform_experiment(experiment_data, data_label, basic_feature_name,
                       redviz_choice=[], experiment_data_out=False):

    # 样本数量，特征数量，原分类数量
    sample_num, feature_num = experiment_data.shape
    label_num = len(np.unique(data_label))

    # 记录相关评估数据
    frame = {
        'tag': [],  # 字符串类型
        'Dunn': [],         # 数值类型
        'ACC': []           # 数值类型
    }
    cnt = 1  # 计数

    # 1.生成属性所有的循环排列
    permutations = generate_cyclic_permutations(feature_num)

    # 2.遍历所有排列
    for per in permutations:
        # 2.1调整属性顺序后的数据
        data = experiment_data[:, per]

        # 2.2属性名称生成
        # feature_name = permutation_to_feature_name(per)
        feature_name = basic_feature_name[per]

        # 2.3radviz可视化坐标计算
        x, y, anchors = radviz_coordinates(data, feature_num)

        # 2.4radviz可视化
        # 有选择的进行可视化
        if cnt in redviz_choice:
            plot_radviz(x, y,
                        anchors,
                        feature_name,
                        data_label,  # 样本的标签
                        segment_color_generate(feature_num),
                        class_colors=class_color_generate(label_num))

        # 2.5 生成聚类二维数据(结合x和y坐标)
        cluster_data = np.column_stack((x,y))

        # 2.6 k-means聚类,获得聚类标签值
        new_label = k_means_cluster(cluster_data, label_num, random_state=42)
        # print(new_label)

        # 2.7 计算dunn指数（传入聚类的二维数据和聚类后的标签）
        dunn = dunn_index(cluster_data, new_label)
        # print(dunn)

        # 2.8 计算准确率acc（传入真实标签和聚类的标签）
        acc = accuracy_rate(data_label, new_label)
        # print(acc)

        # 2.9 放入数据表格的数据
        frame['tag'].append(str(tuple(per)))
        frame['Dunn'].append(dunn)
        frame['ACC'].append(acc)

        cnt += 1

    # 3.创建 DataFrame
    res_table = pd.DataFrame(frame)
    print(res_table)
    if experiment_data_out:
        res_table.to_excel('./results/experiment_output.xlsx', index=False)

    # 4.计算Dunn min，max，average
    dunn_data = frame['Dunn']
    print(f"Dunn min:{'%.4f' % np.min(dunn_data)},max:{'%.4f' % np.max(dunn_data)},"
          f"average:{'%.4f' % np.average(dunn_data)}")

    # 5. 计算acc min，max，average
    acc_data = frame['ACC']
    print(f"ACC min:{'%.4f' % np.min(acc_data)},max:{'%.4f' % np.max(acc_data)}, "
          f"average:{'%.4f' %  np.average(acc_data)}")
