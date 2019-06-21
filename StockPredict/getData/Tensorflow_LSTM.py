import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

start=time.time()

 #——————————————————导入数据——————————————————————
f=open('601988.csv')
df=pd.read_csv(f)
data=np.array(df['high'])#获取最高价
data=data[::-1] #反转
plt.figure()
plt.plot(data)
plt.show()

normalize_data=(data-np.mean(data))/np.std(data)  #标准化
normalize_data=normalize_data[:,np.newaxis]  #增加维度

# 生成训练集
# 设置常量
time_step=4    # 时间步
rnn_unit=10       # hidden layer units
batch_size=64  #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.001         #学习率
train_x,train_y=[],[]   #训练集
for i in range(len(data)-time_step-1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

#——————————————————定义神经网络变量——————————————————
with tf.name_scope('input_layer'):
    X=tf.placeholder(tf.float32, [None,time_step,input_size],name='X_input')    #每批次输入网络的tensor
    Y=tf.placeholder(tf.float32, [None,time_step,output_size],name='Y_input')   #每批次tensor对应的标签
#输入层、输出层权重、偏置
with tf.name_scope('weight'):
    weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
    tf.summary.histogram('hidden_layer/weight',weights['in'])
with tf.name_scope('bias'):
    biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }
    tf.summary.histogram('hidden_layer/bias',biases['in'])



#——————————————————定义神经网络变量——————————————————
def lstm(batch):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.LSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入

    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    with tf.variable_scope('lstm'):
        pred,_=lstm(batch_size)
    #损失函数
    with tf.name_scope('loss'):
        loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
        tf.summary.scalar('loss',loss)
    with tf.name_scope('train'):
        train_op=tf.train.AdamOptimizer(lr).minimize(loss)
        saver=tf.train.Saver(tf.global_variables())
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        lossed=[]
        #重复训练1000次
        for i in range(1000):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #每10步保存一次参数
                if (step+1)%10==0:
                    print(i,loss_)
                    lossed.append(loss_)
                    print("保存模型：",saver.save(sess,'./model/stock.model'))
                step+=1
        plt.figure()
        plt.plot(list(range(len(lossed))), lossed, color='b')
        plt.xlabel("train times")
        plt.ylabel("loss_rate")
        plt.show()

        merged=tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)

train_lstm()


#————————————————预测模型————————————————————
def prediction():
    with tf.variable_scope('lstm', reuse=True):
        pred,_=lstm(1)      #预测时只输入[1,time_step,input_size]的测试数据
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, module_file)
        predict=[]
        #得到之后20个预测结果
        for i in range(len(train_x)-1):
            # 取训练集最后一行为测试样本。shape=[1,time_step,input_size]
            prev_seq = train_x[i]
            #print("用作预测的样本:", i,prev_seq)
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            #print(pred[0])
            predict.append(next_seq[-1])

        predict=np.array(predict)*np.std(data)+np.mean(data)
        print("lenp",len(predict))

        end = time.time()
        print("time cost:", end - start, "s")

        b = []
        for i in range(len(predict)):
            b.append(predict[i][0])
        #计算平均绝对误差MAE
        mae = np.average(np.abs(np.array(b[0:len(b)]) - np.array(data[time_step:len(b)+time_step])))
        #mse = np.sqrt(np.mean((predict[time_step:len(data)]-data[time_step:len(data))**2))
        print("平均绝对误差MAE：",mae)
        #预测未来十天的价格
        predict_ten=[]
        prev_seq2 = train_x[-1]
        for i in range(10):
            next_seq2 = sess.run(pred, feed_dict={X: [prev_seq2]})
            predict_ten.append(next_seq2[-1][0])
            # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq2 = np.vstack((prev_seq2[1:], next_seq2[-1]))
        predict_ten = np.array(predict_ten) * np.std(data) + np.mean(data)
        print(predict_ten)
        #print("预测未来十天的价格",predict[len(predict) - 10:len(predict)])
        for i in range(len(predict_ten)):
            b.append(predict_ten[i])
        #以折线图表示结果
        plt.figure(figsize=(12,4))
        plt.plot(list(range(len(data))), data, color='b',label='history')
        plt.plot(list(range(time_step-1,len(b)+time_step-1)), b, color='r',label='predict',linestyle='--')
        plt.title("history and predict")
        plt.xlabel("time_sequence")
        plt.ylabel("high")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,2))
        plt.plot(data[1046:1053], color='b', label='history')
        plt.plot(predict[1046-time_step-2:1053-time_step-2], color='r', label='predict',linestyle='--')
        plt.xlabel("close time Time-Series")
        plt.ylabel("price")
        plt.legend()
        plt.show()


prediction()

