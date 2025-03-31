# -*- coding: utf-8 -*-
# @File    : test_permutation.py
# @Time    : 2025/3/28 21:49
# @Author  : wuhao
# @Description:
#

from cyclic_permutation import generate_cyclic_permutations, permutation_to_feature_name
import numpy as np

# 示例用法
# n = 4
# cyclic_perms = generate_cyclic_permutations(n)
# print(cyclic_perms)  # 输出：[[0, 1, 2], [0, 2, 1]]

a = np.array([[1,2,3,4], [-1,-2,-3,-4]])
m,n = a.shape

permutations = generate_cyclic_permutations(n)

# 遍历所有排列
for per in permutations:
    print(f"使用排列{per}")
    feature_name = permutation_to_feature_name(per)
    print(f"特征名称顺序:{feature_name}")

    change_a = a[:, per]
    print(f"调整顺序后的数据:")
    print(change_a)


name = np.array(['Dim1', 'Dim2', 'Dim3', 'Dim4'])

index = [1,0,2,3]
print(name[index])