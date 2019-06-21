import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
start=time.time()

# 定义常量
time_step=3
rnn_unit = 10  # hidden layer units
input_size = 5
output_size = 1
lr = 0.0006  # 学习率
# ——————————————————导入数据——————————————————————
f = open('600519.csv')
df = pd.read_csv(f,index_col="trade_date",parse_dates=True).sort_index()  # 读入股票数据
data = df.iloc[:, 1:7].values  # 取第2-7列
print(data)



# 获取训练集
def get_train_data(batch_size=64, train_begin=0, train_end=800):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :5]
        y = normalized_train_data[i:i + time_step,5, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(test_begin=800):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y,last_data = [], [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :5]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, 5]
        test_x.append(x.tolist())
        test_y.extend(y)
    z=normalized_test_data[len(normalized_test_data)-time_step:len(normalized_test_data),:5]
    last_data.append(z.tolist())
    return mean, std, test_x, test_y, last_data

# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm(batch_size=64, train_begin=0, train_end=800):
    with tf.name_scope('input_layer'):
        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size],name='X_input')
        Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size],name='Y_input')
    # 训练样本中前800个样本，每次取20个
    batch_index, train_x, train_y = get_train_data(batch_size,train_begin, train_end)
    print(np.array(train_x).shape)  # 780  20  5
    print(batch_index)
    # 相当于总共780句话，每句话20个字，每个字5个特征（embadding）,对于这些样本每次训练80句话
    with tf.variable_scope("lstm"):
        pred, _ = lstm(X)
        tf.summary.scalar('pred', pred)
    # 损失函数
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        lossed = []
        # 重复训练1000次
        for i in range(3000):
            # 每次进行训练的时候，每个batch训练batch_size个样本
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],Y: train_y[batch_index[step]:batch_index[step + 1]]})
            print(i, loss_)
            lossed.append(loss_)
            if i % 200 == 0:
                print("保存模型：", saver.save(sess, 'model2/stock.model', global_step=i))
    plt.figure()
    plt.plot(list(range(len(lossed))), lossed, color='b')
    plt.xlabel("time_sequence")
    plt.ylabel("loss_rate")
    plt.show()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs2/", sess.graph)

train_lstm()


# ————————————————预测模型————————————————————
def prediction():
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size],name='predict_input')
    mean, std, test_x, test_y, l_data = get_test_data(test_begin=0)
    with tf.variable_scope("lstm", reuse=True):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('./model2')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        print(test_x[-1])
        print(l_data[0])
        #预测后20的价格
        prob2=sess.run(pred,feed_dict={X: [l_data[0]]})
        predict2=prob2.reshape((-1))
        print("kk__",predict2)
        test_predict.extend(predict2)
        print(np.array(predict2)*std[5]+mean[5])
        #参数恢复
        test_y = np.array(test_y) * std[5] + mean[5]
        test_predict = np.array(test_predict) * std[5] + mean[5]
        print("预测数组长度：",len(test_predict))
        print("最后十个预测数据：",test_predict[len(test_predict)-10:-1])
        #测试集的绝对平方误差
        mae = np.average(np.abs(test_predict[:len(test_y)] - test_y[:len(test_y)]))  # 偏差
        print("绝对平方误差MAE:",mae)
        # 以折线图表示结果
        plt.figure(figsize=(12,4))
        plt.plot(list(range(len(test_predict))), test_predict, color='r',label='predict',linestyle='--')
        plt.plot(list(range(len(test_y))), test_y, color='b',label='history')
        plt.title("history and predict")
        plt.xlabel("time_sequence")
        plt.ylabel("high")
        plt.legend()
        plt.show()


prediction()

end=time.time()
print("time cost:",end-start,"s")