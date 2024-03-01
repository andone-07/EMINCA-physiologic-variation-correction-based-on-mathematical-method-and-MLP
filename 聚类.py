# -*- coding: gbk -*- #
import pandas as pd #����pandas��
import numpy as np #����numpy��
import matplotlib.pyplot as plt #����matplotlib��
from sklearn.cluster import KMeans #����sklearn��

X = []
f = open('Aml.csv') #��csv�ļ�
for v in f:
    X.append([float(v.split(",")[0]),float(v.split(",")[2])]) #ѡȡ��0�к͵ڶ������ݣ�ͨ���������ָ���������

#ת��Ϊnumpy array
X = np.array(X)
print(X)

#��ص�����
n_clusters = 2
 
#��ʼ���ú�������
cls = KMeans(n_clusters).fit(X)

#���X��ÿ�����������һ���б�
print(cls.labels_)
 
#��ͼ
markers = ['*','o']#,'+','s','v'] #��״
colors = ['black','red']#,'blue','green','yellow'] #��ɫ
print("����㣺")
for i in range(n_clusters):
    members = cls.labels_ == i   #members�ǲ�������
    plt.scatter(X[members,0],X[members,1],s = 60,marker = markers[i],color = colors[i],alpha=0.5)   #����menbers������ƥ��ĵ�
    print(X[members,0],X[members,1])
#����s����ʶ��״��С marker����ʶ color����ɫ alpha��͸����
plt.title('����') #����
plt.show() #չʾͼƬ
print(X) #��ӡ����