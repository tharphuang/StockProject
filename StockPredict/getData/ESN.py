import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from matplotlib import pyplot as plt
from pyESN import ESN
import time
time_start=time.time()

f=open('000001.csv')
df=pd.read_csv(f)
data=np.array(df['high']).astype('float64')#获取最高价
data=data[::-1] #反转

print(data)

"""
n_inputs: 输入维数
n_outputs: 输出尺寸 
n_reservoir: 储备吃神经元个数
random_state: 随机种子的随机发生器
sparsity: proportion of recurrent weights set to zero 
spectral_radius: 谱半径权重 
noise: noise added to each neuron (regularization) 
"""

n_reservoir= 500
sparsity=0.2
rand_seed=23
radius_set = [0.9,  1,  1.1]
noise_set = [ 0.001, 0.004, 0.006]

radius_set_size  = len(radius_set)
noise_set_size = len(noise_set)

trainlen = 1000
future = 1
futureTotal=54

pred_tot=np.zeros(futureTotal)

loss = np.zeros([radius_set_size, noise_set_size])

#均方误差
def MSE(yhat, y):
    #return np.sqrt(np.mean((yhat.flatten() - y)**2))
    return np.average(np.abs(yhat.flatten() - y ))
'''
#不同的预测步骤
for l in range(radius_set_size):
    rho = radius_set[l]
    for j in range(noise_set_size):
        noise = noise_set[j]
        pred_tot = np.zeros(futureTotal)
        esn = ESN(n_inputs=1,
                  n_outputs=1,
                  n_reservoir=n_reservoir,
                  sparsity=sparsity,
                  random_state=rand_seed,
                  spectral_radius=rho,
                  noise=noise)

        for i in range(0, futureTotal, future):
            pred_training = esn.fit(np.ones(trainlen), data[i:trainlen + i])
            prediction = esn.predict(np.ones(future))
            pred_tot[i:i + future] = prediction[:, 0]

        loss[l, j] = MSE(pred_tot, data[trainlen:trainlen + futureTotal])
        print('rho = ', radius_set[l], ', noise = ', noise_set[j], ', MSE = ', loss[l][j])


#影响因子的组合
plt.figure(figsize=(16,8))
im = plt.imshow(loss.T, vmin=abs(loss).min(), vmax=abs(loss).max(), origin='lower',cmap='PuRd')
plt.xticks(np.linspace(0,radius_set_size-1,radius_set_size), radius_set);
plt.yticks(np.linspace(0,noise_set_size-1, noise_set_size), noise_set);
plt.xlabel(r'$\rho$', fontsize=16); plt.ylabel('noise', fontsize=16);

# im.set_interpolation('bilinear')
cb = plt.colorbar(im);

#影响因子的最优解
minLoss = np.min(loss)
index_min = np.where(loss == minLoss)
index_min
rho_opt = radius_set[int(index_min[0])]
noise_opt = noise_set[int(index_min[1])]
print('最优解的组合:\nspectrum radius = ',rho_opt,'\nnoise = ',noise_opt,'\nMSE = ',minLoss)
'''
#预测步长
spectral_radius = 1
noise = .001
'''
k_set = [1,2,5,10]
k_size = len(k_set)
loss_k = np.zeros(k_size)

for l in range(k_size):
    future = k_set[l]

    #trainlen = 1000
    #futureTotal = 100

    pred_tot = np.zeros(futureTotal)

    esn = ESN(n_inputs=1,
              n_outputs=1,
              n_reservoir=n_reservoir,
              sparsity=sparsity,
              random_state=rand_seed,
              spectral_radius=spectral_radius,
              noise=noise)

    for i in range(0, futureTotal, future):
        pred_training = esn.fit(np.ones(trainlen), data[i:trainlen + i])
        prediction = esn.predict(np.ones(future))
        pred_tot[i:i + future] = prediction[:, 0]

    loss_k[l] = MSE(pred_tot, data[trainlen:trainlen + futureTotal])

plt.figure(figsize=(14,8))
plt.plot(k_set, loss_k,'-ok', alpha=0.6)
plt.title('MSE as a Function of Window Length', fontsize=20)
plt.xlabel('Window Length', fontsize=18)
plt.ylabel('MSE', fontsize=18)
plt.show()
sns.despine()
'''
#最优解预测未来两天的价格

esn = ESN(n_inputs = 1,
      n_outputs = 1,
      n_reservoir = n_reservoir,
      sparsity=sparsity,
      random_state=rand_seed,
      spectral_radius = spectral_radius,
      noise=noise)
for i in range(0,futureTotal,future):
    pred_training = esn.fit(np.ones(trainlen),data[i:trainlen+i])
    prediction = esn.predict(np.ones(future))
    pred_tot[i:i+future] = prediction[:,0]

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=False)
plt.figure(figsize=(16,8))
plt.plot(range(500,trainlen+futureTotal),data[500:trainlen+futureTotal],'b',label="Data", alpha=0.3)
#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
plt.plot(range(trainlen,trainlen+futureTotal),pred_tot,'k',  alpha=0.8, label='ESN')

lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:', linewidth=4)

plt.title(r'Truth and ESN Output', fontsize=25)
plt.xlabel(r'Time', fontsize=20,labelpad=10)
plt.ylabel(r'Price', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
plt.show()
sns.despine()

time_end=time.time()
print("time cost",time_end-time_start,"s")

