import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io

# 加载数据

f=open('000001.csv')
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
time_step=20      # 时间步
rnn_unit=10       # hidden layer units
batch_size=64     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0006         #学习率
train_x,train_y=[],[]   #训练集

for i in range(len(data)-time_step-1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

input_dim = 1
X = tf.placeholder(tf.float32, [None, time_step, input_dim])
Y = tf.placeholder(tf.float32, [None, time_step])


# regression
def ass_rnn(hidden_layer_size=rnn_unit):
    W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
    b = tf.Variable(tf.random_normal([1]), name='b')
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(X)[0], 1, 1])
    out = tf.matmul(outputs, W_repeated) + b
    out = tf.squeeze(out)
    return out


def train_rnn():
    out = ass_rnn()

    loss = tf.reduce_mean(tf.square(out - Y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        sess.run(tf.global_variables_initializer())

        for step in range(100):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
            if step % 10 == 0:
                # 用测试数据评估loss
                print(step, loss_)
        print("保存模型: ", saver.save(sess, './model3/stock.model'))


train_rnn()

def prediction():
    out = ass_rnn()

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        saver.restore(sess, './model3/stock.model')

        prev_seq = train_x[-1]
        predict = []
        for i in range(12):
            next_seq = sess.run(out, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()

prediction()