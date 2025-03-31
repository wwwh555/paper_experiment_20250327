# -*- coding: utf-8 -*-
# @File    : test1.py
# @Time    : 2025/3/27 15:56
# @Author  : wuhao
# @Description:
#

from function.radviz import plot_radviz, radviz_coordinates
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler

sklearn.datasets.load_iris()

# 生成示例数据（4个特征，3个类别）
np.random.seed(42)
n_samples = 100
data = np.random.rand(n_samples, 4)  # 4个特征
features = ['Feature1', 'Feature2', 'Feature3', 'Feature4']

# 生成类别标签（0/1/2）
class_labels = np.random.randint(0, 3, n_samples)

# 自定义颜色方案
anchor_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 锚点颜色
class_colors = ['#FF9999', '#66B2FF', '#99FF99']  # 类别颜色

# 数据归一化
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data)

# 计算坐标并绘图
x, y, anchors = radviz_coordinates(data_norm, features)
plot_radviz(x, y, anchors, features, class_labels,
            anchor_colors=anchor_colors,
            class_colors=class_colors,
            title="Radviz with Class & Feature Colors")