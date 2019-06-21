
import numpy as np
import pandas as pd

class DataLoader():
	"""
	加载数据，并将数据转换为符合LSTM输入的格式
	"""

	def __init__(self, filename, split, cols):
		'''
		filename:数据所在文件名， '.csv'格式文件
		split:训练与测试数据分割变量
		cols:选择data的一列或者多列进行分析，如 Close 和 Volume
		'''
		dataframe = pd.read_csv(filename)
		i_split = int(len(dataframe) * split)
		self.data_train = dataframe.get(cols).values[:i_split]		#选择指定的列 进行分割 得到 未处理的训练数据
		self.data_test = dataframe.get(cols).values[i_split:]
		self.len_train = len(self.data_train)
		self.len_test = len(self.data_test)
		self.len_train_windows = None


	def get_test_data(self, seq_len, normalise):
		'''
		Create x, y test data windows
		Warning: batch method, not generative, make sure you have enough memory to
		load data, otherwise reduce size of the training split.
		'''
		data_windows = []
		for i in range(self.len_test - seq_len):
			data_windows.append(self.data_test[i:i+seq_len]) #每一个元素是长度为seq_len的 list即一个window

		data_windows = np.array(data_windows).astype(float)
		data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

		x = data_windows[:, :-1]
		y = data_windows[:, -1, [0]]

		return x, y

	def get_train_data(self, seq_len, normalise):
		'''
		Create x, y train data windows
		Warning: batch method, not generative, make sure you have enough memory to
		load data, otherwise use generate_training_window() method.
		'''
		data_x = []
		data_y = []
		for i in range(self.len_train - seq_len):
			x, y = self._next_window(i, seq_len, normalise)
			data_x.append(x)
			data_y.append(y)

		return np.array(data_x), np.array(data_y)

	def generate_train_batch(self, seq_len, batch_size, normalise):
		'''
		Yield a generator of training data from filename on given list of cols split for train/test
		'''
		i = 0
		while i < (self.len_train - seq_len):
			x_batch = []
			y_batch = []
			for b in range(batch_size):
				if i >= (self.len_train - seq_len):
					# stop-condition for a smaller final batch if data doesn't divide evenly
					yield np.array(x_batch), np.array(y_batch)
				x, y = self._next_window(i, seq_len, normalise)
				x_batch.append(x)
				y_batch.append(y)
				i += 1
			yield np.array(x_batch), np.array(y_batch)

	def _next_window(self, i, seq_len, normalise):
		'''
		Generates the next data window from the given index location i
		'''
		window = self.data_train[i:i+seq_len]
		window = self.normalise_windows(window, single_window=True)[0] if normalise else window
		x = window[:-1]
		y = window[-1, [0]]  # 最后一行的 0个元素 组成array类型，若是[0,2]则取第0个和第2个元素组成array，[-1, 0]：则是取最后一行第0个元素，
		# 只返回该元素的值[]和()用于索引都是切片操作，所以这里的y即label是 第一列Close列
		return x, y

	def normalise_windows(self, window_data, single_window=False):
		'''
		Normalise window with a base value of zero
		'''
		normalised_data = []
		window_data = [window_data] if single_window else window_data
		for window in window_data:
			normalised_window = []
			for col_i in range(window.shape[1]):
				normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
				normalised_window.append(normalised_col)
			normalised_window = np.array(normalised_window).T  # reshape and transpose array back into original multidimensional format
			normalised_data.append(normalised_window)
		return np.array(normalised_data)