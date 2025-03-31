# -*- coding: utf-8 -*-
# @File    : 350point_experiment.py
# @Time    : 2025/3/30 13:40
# @Author  : wuhao
# @Description:
#

import numpy as np
import pandas as pd

from function.experiment import perform_experiment

'''
1.读取数据
'''
# 1.新特征数据
data_frame = pd.read_csv("./results/350point_new_dim.csv", sep=',')
new_feature_data = data_frame[['Dim1-1', 'Dim1-2', 'Dim2-1', 'Dim2-2', 'Dim3-1', 'Dim3-2', 'Dim3-3',
                                'Dim4-1', 'Dim4-2', 'Dim4-3']].to_numpy()
print(new_feature_data)
print(new_feature_data.shape)

# 2.原数据
data_frame = pd.read_csv("./data/PaperData-软件学报-4DIn12D-350point-withClass3-withXYpointByMDS.csv", sep=',')
data = data_frame[['dim8', 'dim7', 'dim6', 'dim2']].to_numpy()
# 3.读取数据原分类label
data_label = data_frame['Name3Class'].to_numpy()
index = data_label == 'ClassC'
data_label[index] = 1
index = data_label == 'ClassB'
data_label[index] = 2
index = data_label == 'ClassA'
data_label[index] = 3
data_label = np.array(data_label, dtype=int)


'''
2.原数据不升维实验
结果：
Dunn min:0.0204,max:0.0367,average:0.0294
ACC min:0.5971,max:0.9629, average:0.8381


最好可视化效果图：Dunn:0.0367 ACC:0.9543
最坏可视化效果图：Dunn:0.0311 ACC:0.5971
'''
# experiment_data = data
# perform_experiment(experiment_data, data_label, np.array(['Dim1','Dim2','Dim3','Dim4']),
#                    [1,2,3,4,5,6])

'''
3.单一维度升维
维度1升到2维
结果:
Dunn min:0.0146,max:0.0554,average:0.0285
ACC min:0.6771,max:0.9657, average:0.9107

最好可视化效果图：Dunn:0.0554 ACC:0.9629
'''
# experiment_data = np.column_stack((new_feature_data[:,[0,1]], data[:, [1,2,3]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1-1','Dim1-2','Dim2','Dim3','Dim4']),
#                    [8, 11])


'''
3.单一维度升维
维度2升到2维
结果：
Dunn min:0.0081,max:0.0502,average:0.0256
ACC min:0.6629,max:0.9714, average:0.9086

最好可视化效果图：Dunn:0.0502 ACC:0.9571
'''

# experiment_data = np.column_stack((data[:,0], new_feature_data[:,[2,3]], data[:,[2,3]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1','Dim2-1','Dim2-2','Dim3','Dim4']),
#                     [2,12])

'''
3.单一维度升维
维度3升到3维
结果：
Dunn min:0.0149,max:1.1778,average:0.5020
ACC min:0.6514,max:1.0000, average:0.8603

最好可视化效果图：Dunn:0.7172 ACC:1
'''

# experiment_data = np.column_stack((data[:,[0,1]], new_feature_data[:,[4,5,6]], data[:,3]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1','Dim2','Dim3-1','Dim3-2',
#                                                           'Dim3-3','Dim4']), [2,50])

'''
3.单一维度升维
维度4升到3维
结果：
Dunn min:0.0121,max:1.6325,average:0.5110
ACC min:0.6000,max:1.0000, average:0.9912

最好可视化效果图：Dunn:1.6325 ACC:1
'''

# experiment_data = np.column_stack((data[:,[0,1,2]], new_feature_data[:,[7,8,9]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1','Dim2','Dim3','Dim4-1',
#                                                           'Dim4-2','Dim4-3']), [32])


'''
4.两维度组合升维
维度1和维度2升维
结果：
Dunn min:0.0045,max:0.0947,average:0.0254
ACC min:0.6029,max:0.9771, average:0.9253
最优可视化聚类效果：0.0947	0.9771

'''
# experiment_data = np.column_stack((new_feature_data[:, [0,1,2,3]], data[:,[2,3]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1-1','Dim1-2','Dim2-1','Dim2-2',
#                                                           'Dim3','Dim4']), [29])


'''
4.两维度组合升维
维度1和维度3升维
结果：
Dunn min:0.0055,max:1.5212,average:0.5274
ACC min:0.5743,max:1.0000, average:0.8569

最优可视化聚类效果：0.9883	1

'''
# experiment_data = np.column_stack((new_feature_data[:, [0,1]], data[:,1],
#                                    new_feature_data[:,[4,5,6]], data[:,3]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1-1','Dim1-2','Dim2','Dim3-1',
#                                                           'Dim3-2','Dim3-3','Dim4']), [687,691])


'''
4.两维度组合升维
维度1和维度4升维
结果：
Dunn min:0.0049,max:2.2300,average:0.5459
ACC min:0.5914,max:1.0000, average:0.9794

最优可视化聚类效果：2.2300	1

'''
# experiment_data = np.column_stack((new_feature_data[:, [0,1]], data[:,[1,2]],
#                                    new_feature_data[:,[7,8,9]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1-1','Dim1-2','Dim2','Dim3',
#                                                           'Dim4-1','Dim4-2','Dim4-3']), [1, 303])


'''
4.两维度组合升维
维度2和维度3升维
结果：
Dunn min:0.0035,max:1.4797,average:0.5383
ACC min:0.5829,max:1.0000, average:0.8575

最优可视化聚类效果：0.9934	1


'''
# experiment_data = np.column_stack((data[:,[0]], new_feature_data[:, [2,3,4,5,6]],
#                                    data[:,3]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1','Dim2-1','Dim2-2','Dim3-1',
#                                                           'Dim3-2','Dim3-3','Dim4']), [47, 212])


'''
4.两维度组合升维
维度2和维度4升维
结果：
Dunn min:0.0025,max:2.4303,average:0.5347
ACC min:0.5914,max:1.0000, average:0.9803

最优可视化聚类效果：2.4303	1

'''
# experiment_data = np.column_stack((data[:,[0]], new_feature_data[:, [2,3]],
#                                    data[:,2], new_feature_data[:, [7,8,9]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1','Dim2-1','Dim2-2','Dim3',
#                                                           'Dim4-1','Dim4-2','Dim4-3']), [47, 212, 566])


'''
4.两维度组合升维
维度3和维度4升维
结果：
Dunn min:0.0024,max:1.8160,average:0.4986
ACC min:0.4229,max:1.0000, average:0.8561

最优可视化聚类效果：1.8032  	1

'''
# experiment_data = np.column_stack((data[:,[0,1]],new_feature_data[:, [4,5,6,7,8,9]]))
# perform_experiment(experiment_data, data_label, np.array(['Dim1','Dim2','Dim3-1','Dim3-2',
#                                                           'Dim3-3','Dim4-1','Dim4-2','Dim4-3']),
#                                                          [337, 1483])