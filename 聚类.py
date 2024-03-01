# -*- coding: gbk -*- #
import pandas as pd #引用pandas库
import numpy as np #引用numpy库
import matplotlib.pyplot as plt #引用matplotlib库
from sklearn.cluster import KMeans #引用sklearn库

X = []
f = open('Aml.csv') #打开csv文件
for v in f:
    X.append([float(v.split(",")[0]),float(v.split(",")[2])]) #选取第0列和第二列数据，通过“，”分隔，浮点型

#转化为numpy array
X = np.array(X)
print(X)

#类簇的数量
n_clusters = 2
 
#开始调用函数聚类
cls = KMeans(n_clusters).fit(X)

#输出X中每项所属分类的一个列表
print(cls.labels_)
 
#画图
markers = ['*','o']#,'+','s','v'] #形状
colors = ['black','red']#,'blue','green','yellow'] #颜色
print("坐标点：")
for i in range(n_clusters):
    members = cls.labels_ == i   #members是布尔数组
    plt.scatter(X[members,0],X[members,1],s = 60,marker = markers[i],color = colors[i],alpha=0.5)   #画与menbers数组中匹配的点
    print(X[members,0],X[members,1])
#参数s：标识形状大小 marker：标识 color：颜色 alpha：透明度
plt.title('聚类') #标题
plt.show() #展示图片
print(X) #打印数据