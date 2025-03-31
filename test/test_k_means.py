# -*- coding: utf-8 -*-
# @File    : test_k_means.py
# @Time    : 2025/3/28 18:06
# @Author  : wuhao
# @Description:
#

from sklearn.cluster import KMeans
import numpy as np
from evaluate import dunn_index, accuracy_rate

# 创建示例数据
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
Y = np.array([1,0,0,1,0,0])
# 初始化 KMeans 模型，指定聚类数量 k=2
kmeans = KMeans(n_clusters=2, random_state=0)

# 训练模型并预测聚类标签
kmeans.fit(X)
labels = kmeans.labels_

# 获取聚类中心
centers = kmeans.cluster_centers_

print("聚类标签:", labels)
print(type(labels))
print("聚类中心:\n", centers)


print("dunn指数:", dunn_index(X, labels))
print("acc:", accuracy_rate(Y, labels))
# 传入原始数据和对应分类的标签