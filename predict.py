import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import xlwt

def process_data(data):
    global x_list
    line_number = data.shape[0] #行数

    gender = np.array(data.iloc[:, 1:2]) #性别
    gender = np.array(((gender - np.mean(gender)) / (max(gender) - min(gender))).reshape(line_number))

    age = np.array(data.iloc[:, 2:3]) #年龄
    age = np.array(((age - np.mean(age)) / (max(age) - min(age))).reshape(line_number))

    height = np.array(data.iloc[:, 3:4]) #身高
    height = np.array(((height - np.mean(height)) / (max(height) - min(height))).reshape(line_number))

    weight = np.array(data.iloc[:, 4:5]) #体重
    weight = np.array(((weight - np.mean(weight)) / (max(weight) - min(weight))).reshape(line_number))

    x_list = [gender, age, height, weight]
    x = np.vstack((gender, age, height, weight))
    x = np.transpose(x)
    y = np.array(data.iloc[:, 10:11]).reshape(line_number) #因变量数据
    #print(y)
    return x, y

def predict(x, y):
    clf = MLPRegressor(hidden_layer_sizes=(3), activation = 'relu', solver = 'adam', alpha = 0.01, batch_size = 10, 
    learning_rate = 'constant', learning_rate_init = 0.01, power_t = 0.5, max_iter = 1000, shuffle = True, 
    random_state = None, tol = 0.0001, verbose = False, warm_start = False, momentum = 0.9, nesterovs_momentum = True, 
    early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, 
    n_iter_no_change = 10, max_fun = 15000)
    clf.fit(x,y)
    y_pred=clf.predict(x) #预测函数预测并存储预测值
    return y_pred

def pearsonr_score(Yc, x):
    r, p = pearsonr(Yc, x)
    return r, p

data=pd.read_csv('男女总体70%.csv') #导入数据集：与性别无关指标导入男女70%数据；有关指标导入男（女）70%数据 
x, y = process_data(data)
y_pred = predict(x, y)
r2 = r2_score(y,y_pred) #预测值与实际值之间的相关系数
print("相关系数:{:.5f}".format(r2))

Yc=y/y_pred#真实值比预测值
print(Yc)
f = xlwt.Workbook('encoding = utf-8') #设置工作簿编码
sheet1 = f.add_sheet('sheet1',cell_overwrite_ok=True) #创建sheet工作表
for i in range(len(Yc)):
    sheet1.write(i,0,Yc[i]) #写入数据参数对应 行, 列, 值
f.save('Yc.xls') #保存.xls到当前工作目录

num = 0
for i in x_list:
    x_name = ["性别", "年龄", "身高", "体重"]
    r, p = pearsonr_score(Yc, i)
    print("Yc与{xname}的皮尔逊相关系数为：".format(xname = x_name[num]), end = "")
    print("{:.5f}, P值为：{:.5f}".format(r, p))
    num += 1