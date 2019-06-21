import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

df = ts.get_hist_data('000001',start='2017-01-01')
df.head(10)
sz=df.sort_index(axis=0, ascending=True) #对index进行升序排列
sz_return=sz[['p_change']] #选取涨跌幅数据
train=sz_return[0:300] #划分训练集
test=sz_return[300:]   #测试集

#直接用最后一个值作为测试集的预测值
train.index =pd.to_datetime(train.index)#转换时间字符格式以方便作图
test.index =pd.to_datetime(test.index)
dd= np.asarray(train.p_change)  #z转换成向量，以便加入y_hat中
y_hat = test.copy()
y_hat['naive'] = dd[len(dd)-1]
plt.figure(figsize=(10,5))
plt.plot(train.index, train['p_change'], label='Train')
plt.plot(test.index,test['p_change'], label='Test')
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()
#计算RMSE
rms = sqrt(mean_squared_error(test.p_change, y_hat.naive))
print(rms)

#直接用训练集平均值作为测试集的预测值
#Simple Average
y_hat_avg = test.copy() #copy test列表
y_hat_avg['avg_forecast'] = train['p_change'].mean() #求平均值
plt.figure(figsize=(10,5))
plt.plot(train['p_change'], label='Train')
plt.plot(test['p_change'], label='Test')
plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()
rms2 = sqrt(mean_squared_error(test.p_change, y_hat_avg.avg_forecast))
print(rms2)

#直接用移动平均法最后一个值作为测试集的预测值
y_hat_avg = test.copy()
y_hat_avg['moving_avg_forecast'] = train['p_change'].rolling(30).mean().iloc[-1]
#30期的移动平均，最后一个数作为test的预测值
plt.figure(figsize=(10,5))
plt.plot(train['p_change'], label='Train')
plt.plot(test['p_change'], label='Test')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()
rms3 = sqrt(mean_squared_error(test.p_change, y_hat_avg.moving_avg_forecast))
print(rms3)