import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

data=pd.read_csv('男女70%.csv')#导入数据集：与性别无关指标导入男女70%数据；有关指标导入男（女）70%数据
s1 = data.shape[0]

x1_1=data.iloc[:,1:2]#性别
x1_1=np.array(x1_1)#转换为数组
#x1=x1_1.reshape((s1))
x1_1=(x1_1-np.mean(x1_1))/(max(x1_1)-min(x1_1))
x1_1=x1_1.reshape((s1))
x1_1=np.array(x1_1)

x2_1=data.iloc[:,2:3]#年龄
x2_1=np.array(x2_1)
#x2=x2_1.reshape((s1))
x2_1=(x2_1-np.mean(x2_1))/(max(x2_1)-min(x2_1))
x2_1=x2_1.reshape((s1))
x2_1=np.array(x2_1)

x3_1=data.iloc[:,3:4]#身高
x3_1=np.array(x3_1)
#x3=x3_1.reshape((s1))
x3_1=(x3_1-np.mean(x3_1))/(max(x3_1)-min(x3_1))
x3_1=x3_1.reshape((s1))
x3_1=np.array(x3_1)

x4_1=data.iloc[:,4:5]#体重
x4_1=np.array(x4_1)
#x4=x4_1.reshape((s1))
x4_1=(x4_1-np.mean(x4_1))/(max(x4_1)-min(x4_1))
x4_1=x4_1.reshape((s1))
x4_1=np.array(x4_1)

x=np.vstack((x1_1,x2_1,x3_1,x4_1))
x=np.transpose(x)

y=data.iloc[:,10:11]#因变量数据
print(y)#输出因变量数据
clf=MLPRegressor(hidden_layer_sizes=(3), activation='relu',solver='adam', alpha=0.01, batch_size=10, learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
clf.fit(x,y.values.ravel())
y_pred=clf.predict(x)#预测函数预测并存储预测值

r2=r2_score(y,y_pred)#预测值与实际值之间的相关系数
print('相关系数:',r2)

y=np.array(y)#将因变量数据转化为数组
y=y.reshape((s1))# 转换为与y_pred相同维数的变量
#plt.plot(x2_1,y_pred,'o')
#plt.show()
#plt.plot(x2_1,y,'*')
#plt.show()
Yc=y/y_pred#真实值比预测值
print(Yc)

r, p = pearsonr(y, Yc)#真实值与校正值
print(r)
print(p)
r1, p1 = pearsonr(Yc, x1_1)#校正值与性别
print(r1)
print(p1)
r2, p2 = pearsonr(Yc, x2_1)#校正值与年龄
print(r2)
print(p2)
r3, p3 = pearsonr(Yc, x3_1)#校正值与身高
print(r3)
print(p3)
r4, p4 = pearsonr(Yc, x4_1)#校正值与体重
print(r4)
print(p4)

######################################30%数据预测
data2=pd.read_csv('男女30%.csv')
s2= data2.shape[0]

x1_2=data2.iloc[:,1:2]#性别
x1_2=np.array(x1_2)
#x1_2=x1_2.reshape((s2))#转换为一维数组
x1_2=(x1_2-np.mean(x1_2))/(max(x1_2)-min(x1_2))
x1_2=x1_2.reshape((s2))
x1_2=np.array(x1_2)

x2_2=data2.iloc[:,2:3]#年龄
x2_2=np.array(x2_2)
#x2_2=x2_2.reshape((s2))
x2_2=(x2_2-np.mean(x2_2))/(max(x2_2)-min(x2_2))
x2_2=x2_2.reshape((s2))
x2_2=np.array(x2_2)

x3_2=data2.iloc[:,3:4]#身高
x3_2=np.array(x3_2)
#x3_2=x3_2.reshape((s2))
x3_2=(x3_2-np.mean(x3_2))/(max(x3_2)-min(x3_2))
x3_2=x3_2.reshape((s2))
x3_2=np.array(x3_2)

x4_2=data2.iloc[:,4:5]#体重
x4_2=np.array(x4_2)
#x4_2=x4_2.reshape((s2))
x4_2=(x4_2-np.mean(x4_2))/(max(x4_2)-min(x4_2))
x4_2=x4_2.reshape((s2))
x4_2=np.array(x4_2)

x_2=np.vstack((x1_2,x2_2,x3_2,x4_2))
x_2=np.transpose(x_2)
y_pred2=clf.predict(x_2)

y_2=data2.iloc[:,10:11]
y_2=np.array(y_2)
y_2=y_2.reshape((s2))
Yc_2=y_2/y_pred2

r_2, p_2 = pearsonr(y_2, Yc_2)
print(r_2)
print(p_2)
r1_2, p1_2 = pearsonr(Yc_2,x1_2)
print(r1_2)
print(p1_2)
r2_2, p2_2 = pearsonr(Yc_2,x2_2)
print(r2_2)
print(p2_2)
r3_2, p3_2 = pearsonr(Yc_2,x3_2)
print(r3_2)
print(p3_2)
r4_2, p4_2 = pearsonr(Yc_2,x4_2)
print(r4_2)
print(p4_2)