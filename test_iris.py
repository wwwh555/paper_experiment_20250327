# -*- coding: utf-8 -*-
# @File    : test_iris.py
# @Time    : 2025/3/27 16:40
# @Author  : wuhao
# @Description:
#
import sys

from function.experiment import perform_experiment
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from function.calculate import mean_shift, calculated_frequency
from function.radviz import plot_group_bar2

'''
1. 导入数据
'''
# 鸢尾花数据集
iris = sklearn.datasets.load_iris()
# # 属性数据
# iris_data = iris.data
# # 数值标签
# iris_target = iris.target
# # 标签字段
# iris_target_names = iris.target_names
# print(iris_data)
# print(iris_target)
# print(iris_target_names)


# 读取iris数据
iris_dataframe = pd.read_csv("./data/PaperData-软件学报-iris-normalization.csv", sep=',')
# print(iris_dataframe)

# 特征
iris_feature_dataframe = iris_dataframe[['col1', 'col2', 'col3', 'col4']]
iris_feature = iris_feature_dataframe.to_numpy()
sample_num, feature_num = iris_feature.shape  # 样本数量和特征数量
# print(iris_feature)

# label
iris_label_dataframe = iris_dataframe['classId']
iris_label = iris_label_dataframe.to_numpy()
label_num = len(np.unique(iris_label))   # 类别数量
print(iris_label)


'''
2.归一化数据，[0,1]归一化
'''
scaler = MinMaxScaler()
# iris_norm = scaler.fit_transform(iris_feature)
iris_norm = iris_feature
print("iris_norm:")
print(iris_norm)

# 连续2次归一化与一次归一化的数据完全相同
# iris_norm2 = scaler.fit_transform(iris_norm)
# print(iris_norm == iris_norm2)
#
# print(iris_norm)
# print(iris_norm == iris_feature)
# iris_norm = iris_feature  # 原数据已经进行标准化了？


'''
3.求不同维度特征概率值，画原始概率分布直方图
'''
# 求p（所有原点的对应p值）
p = [[] for _ in range(feature_num)]
# for i in range(feature_num):
#     probabilities, centers, width = calculated_frequency(iris_norm[:,i], segment=50)
#     p[i] = calculate_sample_p(iris_norm[:, i], probabilities, width)

# 咱们现在只对那50个概率密度的中点进行漂移？
for i in range(feature_num):
    p[i], centers, width = calculated_frequency(iris_norm[:, i], segment=50)

# 画直方图
# for i in range(feature_num):
#     plot_bar(iris_norm, i)

# print(p[0])
# print(len(p[0]))
# 返回probabilities为当前输入样本对应维度特征的概率值

# 自定义颜色方案
# anchor_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 锚点颜色
# class_colors = ['#FF9999', '#66B2FF', '#99FF99']  # 类别颜色

'''
4.分别对每个维度进行均值漂移，对每个维度进行均值漂移，每个维度的特征空间有2个，构成：(该维度取值，对应概率值)
    改成（区间中点，柱形对应概率值），即对柱状中的50个柱形上边缘中点进行均值漂移
'''
# # 对每个属性列依次进行均值漂移
# for i in range(feature_num):
#     # 1.构建特征空间数据
#     data = np.column_stack( (iris_norm[:, i], p[i]) )  # x轴：属性取值，y轴：对应概率值
#
#     # 2.进行均值漂移,得到响应分组
#     data_shift, group = mean_shift(data, h=0.2)
#     # # 3.计算分组
#     # group = groups(data_shift)
#     print(f"feature{i}, group: {group}")
#
#     # 4,绘制分组的直方图
#     plot_group_bar(iris_norm[:, i], i, group)
# ------------------------------------------------

# 对每个属性列依次进行均值漂移，且不同的分组
groups = [[] for _ in range(feature_num)]
for i in range(feature_num):
    # 1.构建特征空间数据
    data = np.column_stack( (centers, p[i]) )  # x轴：属性取值，y轴：对应概率值

    # 2.进行均值漂移,得到响应分组
    if i == 0:
        h_now = 0.085  # 将维度1切分为3个维度
        # h_now = 0.145
    elif i == 2:
        h_now = 0.17  # 维度3切分维2个维度
        # h_now = 0.145
    else:
        h_now = 0.145

    # h_now = 0.145

    data_shift, group = mean_shift(data, h=h_now)

    groups[i] = group
    # # 3.计算分组
    # group = groups(data_shift)
    print(f"feature{i}, group: {group}")

    # 4.绘制分组的直方图
    # plot_group_bar2(i, p[i], centers, width, group)

# 主动停止程序
# sys.exit()


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
            if iris_norm[j, i] < scope[k]:
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
        f[index] = (iris_norm[:, i])[index]
        if np.sum(f) == 0:  # 这个类别直接去掉
            continue
        new_feature.append(f.tolist())

    ff[i] = np.column_stack(new_feature)  # i特征的拆分特征

new_iris_feature = [[] for i in range(sample_num)]
for i in range(feature_num):
    new_iris_feature = np.hstack((new_iris_feature, ff[i]))

# frame = pd.DataFrame(new_iris_feature, columns=['Dim1-1', 'Dim1-2', 'Dim2-1','Dim2-2','Dim3-1','Dim3-2',
#                                         'Dim3-3', 'Dim4-1','Dim4-2','Dim4-3'])

# frame.to_csv("./results/iris_new_dim.csv", index=True)


# print(new_iris_feature)
# m,n = new_iris_feature.shape
# print(m)
# print(n)
# new_iris_feature 前2、2、3、3分别为原来四个属性升维后的属性

# sys.exit()


'''
进行可视化对比实验
1.原始4个维度数据下进行
传入当前实验数据，以及所有样本原始label，和对应属性特征的标签值

结果：
Dunn min:0.0593,max:0.0720,average:0.0640
ACC min:0.7200,max:0.9067, average:0.8422

图7 a
'''
# perform_experiment(iris_norm, iris_label, np.array(['Dim1', 'Dim2', 'Dim3', 'Dim4']),
#                    [1,2,3,4,5,6])


'''
2.对维度1单维度升到2维
结果：
Dunn min:0.0177,max:0.7258,average:0.3339
ACC min:0.4533,max:0.6800, average:0.6606
明显比原来更差，存在过度分类
'''
# 对维度1升维
# experiment_data = np.column_stack((new_iris_feature[:,[0,1]], iris_norm[:,[1,2,3]]))
# # print(experiment_data)
# perform_experiment(experiment_data, iris_label, np.array(['Dim1-1', 'Dim1-2', 'Dim2', 'Dim3', 'Dim4']),
#                    [15])

'''
3.仅对维度2升到2维
结果：
Dunn min:0.0319,max:0.7371,average:0.2076
ACC min:0.5333,max:0.8733, average:0.6978
'''
# experiment_data = np.column_stack((iris_norm[:,[0]],new_iris_feature[:,[2,3]],iris_norm[:,[2,3]]))
# print(experiment_data)
# print(experiment_data.shape)
# perform_experiment(experiment_data, iris_label, np.array(['Dim1', 'Dim2-1', 'Dim2-2', 'Dim3', 'Dim4']),
#                    [1,8])


'''
4.仅对维度3升到3维，共6个维度
结果：
Dunn min:0.0157,max:1.3122,average:0.4264
ACC min:0.5400,max:0.9133, average:0.8844

'''
# experiment_data = np.column_stack((iris_norm[:,[0,1]], new_iris_feature[:, [4,5,6]], iris_norm[:,3]))
# # print(experiment_data)
# # print(experiment_data.shape)
# perform_experiment(experiment_data, iris_label,
#                    np.array(['Dim1', 'Dim2', 'Dim3-1', 'Dim3-2','Dim3-3', 'Dim4']),
#                    [2,4,12])

'''
5.仅对维度4升到3维
结果：
Dunn min:0.0214,max:1.4789,average:0.3927
ACC min:0.5933,max:0.9867, average:0.9494
'''
# experiment_data = np.column_stack((iris_norm[:,[0,1,2]], new_iris_feature[:, [7,8,9]]))
# print(experiment_data)
# print(experiment_data.shape)
# perform_experiment(experiment_data, iris_label,
#                    np.array(['Dim1', 'Dim2', 'Dim3', 'Dim4-1','Dim4-2', 'Dim4-3']),
#                    [14, 20])



'''
6.将维度3升到2维(h=0.17，维度4升到3维(h=0.145)，共7个维度
结果：
Dunn min:0.0055,max:1.7514,average:0.4971
ACC min:0.5267,max:0.9867, average:0.9568

图7 c
'''
# experiment_data = np.column_stack((iris_norm[:,[0,1]], new_iris_feature[:, [4,5,6,7,8]]))

# 二值化
experiment_data = np.column_stack((iris_norm[:,[0,1]], new_iris_feature[:, [4,5,6,7,8]]))
index = (experiment_data != 0)
experiment_data[index] = 1  # 非0位置全部赋值1

# 设置打印选项：threshold=np.inf 表示打印所有元素
# np.set_printoptions(threshold=np.inf)
# print(experiment_data)
# print(experiment_data.shape)
# perform_experiment(experiment_data, iris_label,
#                    np.array(['Dim1', 'Dim2', 'Dim3-1', 'Dim3-2','Dim4-1', 'Dim4-2', 'Dim4-3']),
#                    [218],True)


# sys.exit()
'''
7.对维度3和维度4都升到3维，共8个维度
结果：
Dunn min:0.0020,max:1.8699,average:0.4130
ACC min:0.5333,max:0.9867, average:0.8811

达不到图7 d 的效果
'''
# experiment_data = np.column_stack((iris_norm[:,[0,1]], new_iris_feature[:, [4,5,6,7,8,9]]))
# print(experiment_data)
# print(experiment_data.shape)
# perform_experiment(experiment_data, iris_label,
#                    np.array(['Dim1','Dim2','Dim3-1','Dim3-2','Dim3-3','Dim4-1','Dim4-2','Dim4-3']))


'''
8.将维度1(h=0.085)和维度4(0.145)都升到3维，共8个维度
结果：
Dunn min:0.0062,max:1.1904,average:0.3102
ACC min:0.3667,max:0.9867, average:0.6160

图7d
'''
experiment_data = np.column_stack((new_iris_feature[:, [0,1,2]],iris_norm[:,[1,2]], new_iris_feature[:, [7,8,9]]))
# 设置打印选项：threshold=np.inf 表示打印所有元素
# np.set_printoptions(threshold=np.inf)
# print(experiment_data)
# print(experiment_data.shape)
perform_experiment(experiment_data, iris_label,
                   np.array(['Dim1-1','Dim1-2','Dim1-3','Dim2','Dim3','Dim4-1','Dim4-2','Dim4-3']),
                   [53,1095, 1304,2564,3121])


