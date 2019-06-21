import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from matplotlib import pyplot as plt
from pyESN import ESN


f=open('000001.csv')
df=pd.read_csv(f)
data=np.array(df['change']).astype('float64')
close=np.array(df['close']).astype('float64')
data=data[::-1] #反转
close=close[::-1]
print(close[1045:-1])
print(data[1045:-1])
print(len(close))
'''
ave=np.average(data)
print(ave)
plt.figure(figsize=(10,2))
plt.plot(data)
plt.xlabel('Time')
plt.ylabel('Change')
plt.title("Daily Income")
plt.show()
'''
predict=[]
predic=[]
average=[]
for k in range(len(data)-5):
    average.append(np.average(data[k:k+5]))
print(average[len(data)-13:-1])
print(len(average))

for k in range(len(average)):
    predict.append(close[k+5]+average[k])
    print(close[k+5],average[k])
print(predict[1040:-1])
print(len(predict))


plt.figure(figsize=(10,2))
plt.plot(close[1046:-1],color='blue',label='True')
plt.plot(predict[1040:-1],color='red',linestyle='--',label='predict')
plt.xlabel('close time Time-Series')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(close[506:-1],color='blue',label='True')
plt.plot(predict[500:-1],color='red',linestyle='--',label='predict')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()




