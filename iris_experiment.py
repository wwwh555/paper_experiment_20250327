# -*- coding: utf-8 -*-
# @File    : iris_experiment.py
# @Time    : 2025/3/30 14:39
# @Author  : wuhao
# @Description:
#

import numpy as np
import pandas as pd

from function.experiment import perform_experiment

'''
0.导入数据
'''
# 1.原数据
data_frame = pd.read_csv("./data/PaperData-软件学报-iris-normalization.csv", sep=',')
data = data_frame[['col1','col2','col3','col4']].to_numpy()
# 2.数据标签
data_label = data_frame['classId'].to_numpy()
# 3.升维后的数据
data_frame = pd.read_csv("./results/iris_new_dim.csv", sep=',')
new_feature_data = data_frame[['Dim1-1','Dim1-2','Dim2-1','Dim2-2','Dim3-1',
                                'Dim3-2','Dim3-3','Dim4-1','Dim4-2','Dim4-3']].to_numpy()



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
# experiment_data = np.column_stack((iris_norm[:,[0,1]], new_iris_feature[:, [5,6,7,8,9]]))

# 二值化
# experiment_data = np.column_stack((iris_norm[:,[0,1]], new_iris_feature[:, [5,6,7,8,9]]))
# index = (experiment_data != 0)
# experiment_data[index] = 1  # 非0位置全部赋值1

# 设置打印选项：threshold=np.inf 表示打印所有元素
# np.set_printoptions(threshold=np.inf)
# print(experiment_data)
# print(experiment_data.shape)
# perform_experiment(experiment_data, iris_label,
#                    np.array(['Dim1', 'Dim2', 'Dim3-1', 'Dim3-2','Dim4-1', 'Dim4-2', 'Dim4-3']),
#                    [1,2,3,5,10,200])


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
# experiment_data = np.column_stack((new_iris_feature[:, [0,1,2]],iris_norm[:,[1,2]], new_iris_feature[:, [8,9,10]]))
# 设置打印选项：threshold=np.inf 表示打印所有元素
# np.set_printoptions(threshold=np.inf)
# print(experiment_data)
# print(experiment_data.shape)
# perform_experiment(experiment_data, iris_label,
#                    np.array(['Dim1-1','Dim1-2','Dim1-3','Dim2','Dim3','Dim4-1','Dim4-2','Dim4-3']))


'''
4.组合升维实验
维度1和维度2升维
结果：
Dunn min:0.0117,max:0.9149,average:0.2829
ACC min:0.4800,max:0.7533, average:0.6683

最佳可视化聚类效果：0.9149	0.64

'''

# experiment_data = np.column_stack((new_feature_data[:, [0,1,2,3]], data[:,[2,3]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1-1', 'Dim1-2', 'Dim2-1', 'Dim2-2',
#                                                           'Dim3', 'Dim4']), [1,88])


'''
4.组合升维实验
维度1和维度3升维
结果：
Dunn min:0.0084,max:0.9289,average:0.3320
ACC min:0.5267,max:0.9267, average:0.7647

最佳可视化聚类效果：
0.028620557	0.9267
# 0.928937446	0.58
'''

# experiment_data = np.column_stack((new_feature_data[:, [0,1]], data[:,[1]], new_feature_data[:,[4,5,6]], data[:,3]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1-1', 'Dim1-2', 'Dim2', 'Dim3-1',
#                                                           'Dim3-2', 'Dim3-3', 'Dim4']), [288,541])


'''
4.组合升维实验
维度1和维度4升维
结果：
Dunn min:0.0115,max:0.8637,average:0.3230
ACC min:0.5333,max:0.9867, average:0.7835

最佳可视化聚类效果：
0.357269822	0.9867
# 0.863684735	0.5667


'''

# experiment_data = np.column_stack((new_feature_data[:, [0,1]], data[:,[1,2]], new_feature_data[:,[7,8,9]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1-1', 'Dim1-2', 'Dim2', 'Dim3',
#                                                           'Dim4-1', 'Dim4-2', 'Dim4-3']), [70,524])


'''
4.组合升维实验
维度2和维度3升维
结果：
Dunn min:0.0114,max:2.3528,average:0.5818
    ACC min:0.5333,max:0.9133, average:0.7956

最佳可视化聚类效果：
2.352775998	0.8667
# 1.036019671	0.9133


'''

# experiment_data = np.column_stack((data[:,[0]], new_feature_data[:,[2,3,4,5,6]], data[:,3]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1', 'Dim2-1', 'Dim2-2', 'Dim3-1',
#                                                           'Dim3-2', 'Dim3-3', 'Dim4']), [55,122])


'''
4.组合升维实验
维度2和维度4升维
结果：
Dunn min:0.0190,max:1.6179,average:0.5286
ACC min:0.5267,max:0.9867, average:0.8335

最佳可视化聚类效果：
# 0.843684527	0.9867
1.617911471	    0.9400

'''

# experiment_data = np.column_stack((data[:,[0]], new_feature_data[:,[2,3]], data[:,2], new_feature_data[:,[7,8,9]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1', 'Dim2-1', 'Dim2-2', 'Dim3',
#                                                           'Dim4-1', 'Dim4-2', 'Dim4-3']), [129,346])



'''
4.组合升维实验
维度3和维度4升维
结果：
Dunn min:0.0020,max:1.8699,average:0.4130
ACC min:0.5333,max:0.9867, average:0.8811

最佳可视化聚类效果：

1.613070674	0.9867
# 1.869925862	0.9

'''

experiment_data = np.column_stack((data[:,[0,1]], new_feature_data[:,[4,5,6,7,8,9]]))
perform_experiment(experiment_data, data_label, np.array(['Dim1', 'Dim2', 'Dim3-1', 'Dim3-2',
                                                          'Dim3-3', 'Dim4-1', 'Dim4-2', 'Dim4-3']),
                                                        [688,2789])