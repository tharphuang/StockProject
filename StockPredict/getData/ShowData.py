import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f=open('000001.csv')
df=pd.read_csv(f)
high=np.array(df['high'])#获取最高价
high=high[::-1] #反转
normalize_data=(high-np.mean(high))/np.std(high)
print(np.mean(high),np.std(high))
plt.figure(figsize=(10,4))
plt.plot(high,label='high',linestyle='--')
#plt.plot(open_,label='open',linestyle=':')
#plt.plot(low,label='low',linestyle='-.')
#plt.plot(close,label='close',linestyle=':',marker="*")
#plt.plot(pre_close,label='pre_close',linestyle=':',marker="o")
plt.plot(normalize_data,label='normalize_data')
plt.title("Z-Score Normalizing")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

f2=open('600519.csv')
df2=pd.read_csv(f2)
data2_high=np.array(df2['high'])#获取最高价
data2_high=data2_high[::-1] #反转
plt.figure(figsize=(10,2))
plt.plot(data2_high,color='green',label='600519')
plt.plot(high,color='red',label='000001')
plt.title("The Trend of Stock Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

