# -*- coding: utf-8 -*-
# @File    : test_matplotlib.py
# @Time    : 2025/3/28 11:45
# @Author  : wuhao
# @Description:
#


import matplotlib.pyplot as plt

plt.figure(1)

plt.plot([1,2,3],[4,5,6])

plt.plot([2,3,4],[5,6,4], color='red')

plt.legend()
plt.show()
