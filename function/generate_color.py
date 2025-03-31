# -*- coding: utf-8 -*-
# @File    : generate_color.py
# @Time    : 2025/3/28 22:07
# @Author  : wuhao
# @Description:
#   颜色函数


def segment_color_generate(n):
    segment_colors = ['#058305', '#5C02A1', '#c2a875','#1933A1','#A30C15','#2B83B1', '#E2E337',
                      '#8d62d5',
                      '#C780FF', '#8084FF']  # 锚点颜色
    return segment_colors[0:n]



def class_color_generate(n):
    class_colors = ['#FF9999', '#66B2FF', '#99FF99','#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # 类别颜色

    # 直接切片
    return class_colors[0:n]