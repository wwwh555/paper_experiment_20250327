# -*- coding: utf-8 -*-
# @File    : cyclic_permutation.py
# @Time    : 2025/3/28 21:54
# @Author  : wuhao
# @Description:
#   生成排列函数

import itertools

'''
生成0到n-1的所有循环排列
'''
def generate_cyclic_permutations(n):
    if n <= 0:
        return []
    rest = list(range(1, n))
    perms = itertools.permutations(rest)
    return [[0] + list(p) for p in perms]


'''
将排列转为对应的特征字符串构成的list/ndarray
'''
def permutation_to_feature_name(permutation):

    feature_name = []

    for p in permutation:

        feature_name.append("Dim"+str(p))

    return feature_name

# def permutation_to_str(permutation):
#
#
#
#     for p in permutation:
#
#         str
#
#     return feature_name