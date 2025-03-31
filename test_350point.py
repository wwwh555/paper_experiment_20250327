# -*- coding: utf-8 -*-
# @File    : test_350point.py
# @Time    : 2025/3/30 10:29
# @Author  : wuhao
# @Description:
#
import sys

import numpy as np
import pandas as pd
from function.calculate import calculated_frequency, mean_shift, calculate_sample_p
from function.radviz import plot_group_bar, plot_group_bar2, plot_bar, plot_group_bar3

'''
1.导入数据
'''
#1.读取csv文件
data_frame = pd.read_csv("./data/PaperData-软件学报-4DIn12D-350point-withClass3-withXYpointByMDS.csv", sep=',')
# print(data_frame)
#2.读取数据点（4个特征）
data = data_frame[['dim8', 'dim7', 'dim6', 'dim2']].to_numpy()
sample_num, feature_num = data.shape
# print(data)
#3.读取数据原分类label
data_label = data_frame['Name3Class'].to_numpy()
# print(data_label)
#4.将字符串的label转为数字
index = data_label == 'ClassC'
data_label[index] = 1
index = data_label == 'ClassB'
data_label[index] = 2
index = data_label == 'ClassA'
data_label[index] = 3
# print(data_label)
data_label = np.array(data_label, dtype=int)
# print(data_label.dtype)


'''
2.进行标准化?
应该不用
'''

'''
3.画特征的原直方图
'''

# for i in range(feature_num):
#     plot_bar(data, i, 50)


'''
4.均值漂移对直方图进行分段
'''
groups = [[] for _ in range(feature_num)]
# 50个数据点均值漂移
for i in range(feature_num):
    #1.计算50个直方图中心的概率分布
    prob,centers,width = calculated_frequency(data[:, i], 50)

    # 把prob为0的柱形去掉
    if i == 3:
        index = (prob == 0)
        prob = np.delete(prob, index)
        centers = np.delete(centers, index)

    #2.构成均值漂移特征空间
    mean_shift_data = np.column_stack((centers, prob))

    if i < 2:
        h = 0.2
    elif i == 2:
        h = 0.14
    else:
        h = 0.15

    #3.进行均值漂移(返回新的位置以及对应数据点的分组)
    final_data, group = mean_shift(mean_shift_data, h)

    if i != 3:
        groups[i] = group
    else:
        new_group = np.zeros(50)
        num = len(np.unique(group))
        left = 0
        for j in range(num-1):
            # 找到第j+2组第一个下标
            index = np.argmax(group == j+2)
            right = int(centers[index] // width)

            new_group[range(left, right)] = j+1

            left = right
        new_group[range(left, 50)] = num
        # print(new_group)
        groups[i] = new_group


    #4.
    if i == 3:
        plot_group_bar3(i, prob, centers, width, group)
    else:
        plot_group_bar2(i, prob, centers, width, group)

sys.exit()

# 原数据点均值漂移
# for i in range(2,3):
#     #1.计算50个直方图中心的概率分布
#     prob,centers,width = calculated_frequency(data[:, i], 50)
#     #2.通过原数据点构成特征空间
#     p = calculate_sample_p(data[:, i], prob, width)
#     mean_shift_data = np.column_stack((data[:,i], p))
#
#     if i < 2:
#         h = 0.2
#     elif i == 2:
#         h = 0.176
#     else:
#         h = 0.18
#
#     #3.进行均值漂移(返回新的位置以及对应数据点的分组)
#     final_data, group = mean_shift(mean_shift_data, h)
#
#     #4.画直方图
#     plot_group_bar(data[:, i], i, group)

'''
5.维度切分
'''
ff = {}
for i in range(feature_num):
    print(f"feature:{i} 进行维度切分")

    # 当且特征样本分组记录
    sample_group = np.zeros(sample_num)

    # 当前分组
    now_group = groups[i]
    # 当前分组数
    group_num = len(np.unique(now_group))
    # 找分组的临界值
    scope = np.zeros(group_num-1)  # range的大小为组数-1
    for j in range(len(scope)):
        # 从第二个分组开始第一个取值的下标
        index = np.argmax(now_group == j+2)
        scope[j] = width * index  # 临界值赋值

    # 遍历所有样本，为他们附上组别
    for j in range(sample_num):
        for k in range(len(scope)):
            if data[j, i] < scope[k]:
                sample_group[j] = k+1  # 类别
                break
        if sample_group[j] == 0:
            sample_group[j] = len(scope) + 1  # 最大的分组


    # 新的特征记录
    new_feature = []
    for j in range(group_num):
        f = np.zeros(sample_num)
        # 分组索引
        index = (sample_group == j+1)

        # 对应索引位置赋值，其他为0
        f[index] = (data[:, i])[index]
        if np.sum(f) == 0:  # 这个类别直接去掉
            continue
        new_feature.append(f.tolist())

    ff[i] = np.column_stack(new_feature)  # i特征的拆分特征

new_iris_feature = [[] for i in range(sample_num)]
for i in range(feature_num):
    new_iris_feature = np.hstack((new_iris_feature, ff[i]))
print(new_iris_feature)
print(new_iris_feature.shape)

'''
将所有特征升维后的数据导出excel
'''

pd_frame = pd.DataFrame(new_iris_feature, columns=['Dim1-1', 'Dim1-2', 'Dim2-1', 'Dim2-2', 'Dim3-1', 'Dim3-2', 'Dim3-3',
                                'Dim4-1', 'Dim4-2', 'Dim4-3'])
# print(pd_frame)
pd_frame.to_csv('./results/350point_new_dim.csv', index=True)