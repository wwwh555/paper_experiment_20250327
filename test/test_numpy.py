# -*- coding: utf-8 -*-
# @File    : test_numpy.py
# @Time    : 2025/3/27 17:38
# @Author  : wuhao
# @Description:
#

import numpy as np

array = np.array([[1,2,3],[4,5,6]], dtype=int)
print(array)
print(type(array))


# print(array[:,1])

print(np.linspace(0, 1, 11))


print(np.linspace((0,1),11))


zeros = np.zeros(50)
print(len(zeros))

for i in range(10):
    print("i=",i)
# print(range(10))/



# p, edge, width = calculated_frequency(feature, 50)
#
# print(p)
# print(edge)
# print(width)

a = np.array([1,2,3,4])
b = np.array([-1,-2,-3,-4])

cc = np.array([[1,2,3] + [4,5,6]])
print(cc)

k = np.array([])

a = np.array([1,1])
b = np.array([1,1])
print("aa ",np.linalg.norm(a-np.array([0,0])))
print(np.linalg.norm(a-b))
# print(np.linalg.norm([1,1]-[1,1]))

e = np.empty(10)
print(e)


print([True for _ in range(3)])

a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.column_stack( (a, b) )
print(c)


a = np.array([0.1, 0.1, 0.15, 0.4, 0.4, 0.5])
group = [0, 0, 0, 1, 1, 1]
res = {}
groups = np.unique(group)

for g in groups:
    print(g == group)
    mask = (group == g)
    res[g] = a[mask] # 存储分组结果

for a in groups:
    print(res[a])


print( 3 // 2)


a = np.array([1,2])
b = np.array([3,3])

print( (b-a)**2 )
print(np.sum((a-b)**2))

print(np.array(range(1,3)))


a = np.array([[1,2,3], ['a', 'b', 'c']])
b = np.array([[11,22,33], [111,222,333]])
c = np.array([9,9,9])

a = np.column_stack(a)
b = np.column_stack(b)
print(np.hstack((a,b)))
# print(np.concatenate((a,b,c.T), axis=0))


a = np.array([1,1,1,1,1,2,2,2,2,2])
b = np.array([1,2,3,4,5,6,7,8,9,10])
c = np.zeros(10)

index = (a == 1)
print(index)
print(b[index])
c[index] = b[index]
print(c)

group = np.array([1,2,2,3,3])
print(np.argmax(group == 2))

a = [1,2,3]
b = [3,2,3]
a.append(b)
print(a)

print(np.min([1,2,3,4]))
print(np.average([1,2,3,4]))


data = np.array([[0,0,0.3],
			    [0.2,0,0.4],
			   [0,0.83,0]])

index = data != 0
print(index)
data[index] = 1
print(data)


a = np.array([1,2,3,0,0,1])
index = a == 0
a = np.delete(a, index)
print(a)
