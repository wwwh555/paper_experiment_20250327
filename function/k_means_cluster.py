# -*- coding: utf-8 -*-
# @File    : k_means_cluster.py
# @Time    : 2025/3/28 18:04
# @Author  : wuhao
# @Description:
#   k-means聚类函数

from sklearn.cluster import KMeans
from sklearn.cluster import k_means

def k_means_cluster(data, n_cluster, random_state):

    # 1.创建对象，设置聚类数量k，以及随机数种子
    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state)

    # kmeans = k_means(n_clusters=n_cluster, random_state=random_state)
    # kmeans.fit(data)
    #
    # return kmeans.labels_

    # 2.训练并预测
    kmeans.fit(data)
    # 3.获取聚类标签, labels 是 ndarray类型
    labels = kmeans.labels_

    return labels