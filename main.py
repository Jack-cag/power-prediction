#######################调用第三方库#############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
########################绘制出一列数据的信息####################################
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 读取 Excel 文件
data = pd.read_excel('C:\\Users\\jack\Desktop\\pythonProject_test\\data_file.xlsx')
time_index = pd.date_range(start='2022-01-01', periods=96, freq='15min')  # 生成一天时间范围内每隔15分钟的时间索引
print(data)
# 选择你感兴趣的那一行数据，假设你想绘制第一行数据
row_data = data.iloc[5,1:]
# 提取 x 和 y 数据
y_data = data.iloc[5,1:].values

df = pd.DataFrame(y_data, index=time_index, columns=['Column 1'])  # 使用整个数组作为数据


# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(df.index, df['Column 1'], marker='o', linestyle='-')

# 添加标题和标签
plt.title('Line Plot of Row Data')
plt.xlabel('X')
plt.ylabel('Y')

# 添加图例
# plt.legend()

# 自动格式化时间轴
plt.gcf().autofmt_xdate()

# 显示图形
plt.show()
'''
##################################绘制出多列数据的信息#########################
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 读取 Excel 文件
data = pd.read_excel('C:\\Users\\jack\Desktop\\pythonProject_test\\data_file.xlsx')
time_index = pd.date_range(start='2022-01-01', periods=96, freq='15min')  # 生成一天时间范围内每隔15分钟的时间索引
print(data)
# 选择你感兴趣的那一行数据，假设你想绘制第一行数据
row_data = data.iloc[5,1:]
# 提取 x 和 y 数据
y_data = data.iloc[0:7,1:].values

df = pd.DataFrame(y_data.T, index=time_index, columns=[f'Column {i+1}'for i in range(7)])  # 使用整个数组作为数据

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制折线图
for column in df.columns:
    plt.plot(df.index, df[column], marker='o', linestyle='-', label=column)

# 添加标题和标签
plt.title('Line Plot of Row Data')
plt.xlabel('X')
plt.ylabel('Y')

# 添加图例
plt.legend()

# 自动格式化时间轴
plt.gcf().autofmt_xdate()

# 显示图形
plt.show()
'''
##################################绘制出连续一周的信息#########################
'''
# 提取数据并转置
y_data = data.iloc[0:7, 1:].values

# 展平数组并连接每行的数据
y_data = np.concatenate(y_data).T

time_index = pd.date_range(start='2022-01-01', periods=672, freq='15min')  # 生成一周时间范围内每隔15分钟的时间索引

df = pd.DataFrame(y_data, index=time_index, columns=['Week 1'])  # 使用整个数组作为数据

# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(df.index, df['Week 1'], marker='o', linestyle='-')

# 添加标题和标签
plt.title('Weekly Load Variation Chart')
plt.xlabel('Weekly time/h')
plt.ylabel('Power/KW')

# 添加图例
# plt.legend()

# 自动格式化时间轴
plt.gcf().autofmt_xdate()

# 显示图形
plt.show()
'''
##################################绘制出连续两周的信息#########################
'''
# 提取数据并转置
y_data_flattened = data.iloc[0:7*2, 1:].values

# 展平数组并连接每行的数据
y_data_flattened = np.concatenate(y_data_flattened).T

time_index = pd.date_range(start='2009-01-01', periods=672*2, freq='15min')  # 生成一周时间范围内每隔15分钟的时间索引

df = pd.DataFrame(y_data_flattened, index=time_index, columns=['Week 1'])  # 使用整个数组作为数据

# 创建一个新的图形
plt.figure(figsize=(20, 6))

# 绘制折线图
plt.plot(df.index, df['Week 1'], marker='o', linestyle='-')

# 添加标题和标签
plt.title('Weekly Load Variation Chart')
plt.xlabel('Weekly time/h')
plt.ylabel('Power/KW')

# 添加图例
# plt.legend()

# 自动格式化时间轴
plt.gcf().autofmt_xdate()

# 显示图形
plt.show()
'''
#######################将图像的绘制封装成为一个函数##############################
#参数1：为文件所在路径； 参数2：起始行； 参数3：终止行： 参数4：起始列： 参数5：终止列
#注excel数据的第一列为表示，所以参数0列对应第二列
#函数功能：给定数据范围，能够根据范围绘制负荷曲线
#
def plot_data_from_excel(file_path, start_row, end_row, start_col, end_col):
    # 读取 Excel 文件
    data = pd.read_excel(file_path)
    time_index = pd.date_range(start='2022-01-01', periods=(end_col - start_col), freq='15min')  # 生成一天时间范围内每隔15分钟的时间索引
    print(data)

    # 提取 x 和 y 数据
    y_data = data.iloc[start_row:end_row, (start_col+1):(end_col+1)].values.T

    # 创建 DataFrame
    df = pd.DataFrame(y_data, index=time_index, columns=[f'2022-01-{i + 1}' for i in range(end_row - start_row)])

    # 创建一个新的图形
    plt.figure(figsize=(10, 6))

    # 绘制折线图
    for column in df.columns:
        plt.plot(df.index, df[column], marker='o', linestyle='-', label=column)

    # 添加标题和标签
    plt.title('Daily Load Variation Chart')
    plt.xlabel('Daily time/h')
    plt.ylabel('Power/KW')

    # 添加图例
    plt.legend()

    # 自动格式化时间轴
    plt.gcf().autofmt_xdate()
    plt.show()
plot_data_from_excel('C:\\Users\\jack\Desktop\\pythonProject_test\\data_file.xlsx',0,5,0,96)
#%%###############################负荷数据预处理：异常值处理#################################
#提取表格1
Load= pd.read_excel('C:\\Users\\jack\Desktop\\pythonProject_test\\data_file.xlsx',sheet_name=0)
print(Load)
# 计算四分位数和四分位距，并处理异常值
for column in Load.columns[1:]:
    Q1 = np.percentile(Load[column], 25)
    Q3 = np.percentile(Load[column], 75)
    IQR = Q3 - Q1

    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    outliers = np.where((Load[column] > upper_limit) | (Load[column] < lower_limit))
    Load.loc[(Load[column] > upper_limit) | (Load[column] < lower_limit), column] = np.nan

# 对缺失值进行插值处理
Load.interpolate(method='linear', inplace=True)
for column in Load.columns[1:]:
    Load.loc[2023:,column]=np.nan
print(Load)
#%%######################参数进行数据预处理：缺失值及异常值处理################################
#提取表格2
Weather= pd.read_excel('C:\\Users\\jack\Desktop\\pythonProject_test\\data_file.xlsx',sheet_name=1)
print(Weather)
Weather.interpolate(method='linear', inplace=True)

# 计算四分位数和四分位距，并处理异常值
for column in Weather.columns[1:3]:
    Q1 = np.percentile(Weather[column], 25)
    Q3 = np.percentile(Weather[column], 75)
    IQR = Q3 - Q1

    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    outliers = np.where((Weather[column] > upper_limit) | (Weather[column] < lower_limit))
    print(f"{column}异常值索引：", outliers[0])
    print(f"{column}异常值：", Weather[column].values[outliers])
    Weather.loc[(Weather[column] > upper_limit) | (Weather[column] < lower_limit), column] = np.nan

# 对缺失值进行插值处理
Weather.interpolate(method='linear', inplace=True)

print(Weather)
#%%################################对VMD模型进行测试:构建测试序列#################################################
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:50:28 2019

@author: Vinícius Rezende Carvalho

Test script for Variational Mode Decomposition (Dragomiretskiy and Zosso, 2014)
Original paper:
Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’,
IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
original MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
"""

#from __future__ import division# if python 2

# Time Domain 0 to T
T = 1344
fs = 1/T
t = np.arange(1,T+1)/T
freqs = 2*np.pi*(t-0.5-fs)/(fs)

# center frequencies of components
f_1 = 2
f_2 = 24
f_3 = 288

# modes
v_1 = (np.cos(2*np.pi*f_1*t))
v_2 = 1/4*(np.cos(2*np.pi*f_2*t))
v_3 = 1/16*(np.cos(2*np.pi*f_3*t))

#
#% for visualization purposes
fsub = {1:v_1,2:v_2,3:v_3}
wsub = {1:2*np.pi*f_1,2:2*np.pi*f_2,3:2*np.pi*f_3}

# composite signal, including noise

f = v_1 + v_2 + v_3 + 0.1*np.random.randn(v_1.size)
f_hat = np.fft.fftshift((np.fft.fft(y_data_flattened)))

# some sample parameters for VMD
alpha = 2000       # moderate bandwidth constraint
tau = 0.            # noise-tolerance (no strict fidelity enforcement)
K = 3              # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7


# Run actual VMD code

#u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

#%%#####################对VMD模型进行测试：将处理后的模型进行可视化######################################
# Simple Visualization of decomposed modes
"""
plt.figure(figsize=(20, 6))
plt.plot(f)
plt.figure()
plt.plot(u.T)
plt.title('Decomposed modes')


# For convenience here: Order omegas increasingly and reindex u/u_hat
sortIndex = np.argsort(omega[-1,:])
omega = omega[:,sortIndex]
u_hat = u_hat[:,sortIndex]
u = u[sortIndex,:]
linestyles = ['b', 'g', 'm', 'c', 'c', 'r', 'k']

fig1 = plt.figure()
plt.subplot(411)
plt.plot(t,f)
plt.xlim((0,1))
for key, value in fsub.items():
    plt.subplot(4,1,key+1)
    plt.plot(t,value)
fig1.suptitle('Original input signal and its components')


fig2 = plt.figure()
plt.loglog(freqs[T//2:], abs(f_hat[T//2:]))
plt.xlim(np.array([1,T/2])*np.pi*2)
ax = plt.gca()
ax.grid(which='major', axis='both', linestyle='--')
fig2.suptitle('Input signal spectrum')


fig3 = plt.figure()
for k in range(K):
    plt.semilogx(2*np.pi/fs*omega[:,k], np.arange(1,omega.shape[0]+1), linestyles[k])
fig3.suptitle('Evolution of center frequencies omega')


fig4 = plt.figure()
plt.loglog(freqs[T//2:], abs(f_hat[T//2:]), 'k:')
plt.xlim(np.array([1, T//2])*np.pi*2)
for k in range(K):
    plt.loglog(freqs[T//2:], abs(u_hat[T//2:,k]), linestyles[k])
fig4.suptitle('Spectral decomposition')
plt.legend(['Original','1st component','2nd component','3rd component'])


fig4 = plt.figure()

for k in range(K):
    plt.subplot(3,1,k+1)
    plt.plot(t,u[k,:], linestyles[k])
    plt.plot(t, fsub[k+1], 'k:')

    plt.xlim((0,1))
    plt.title('Reconstructed mode %d'%(k+1))
plt.show()
"""
#%%#######################对实际数据进行测试#########################################
# 提取数据并转置
u, u_hat, omega = VMD(y_data_flattened, alpha, tau, K, DC, init, tol)
plt.figure(figsize=(20, 6))
plt.plot(y_data_flattened)
plt.figure()
plt.plot(u.T)
plt.title('Decomposed modes')


# For convenience here: Order omegas increasingly and reindex u/u_hat
sortIndex = np.argsort(omega[-1,:])
omega = omega[:,sortIndex]
u_hat = u_hat[:,sortIndex]
u = u[sortIndex,:]
linestyles = ['b', 'g', 'm', 'c', 'c', 'r', 'k']

fig1 = plt.figure()
plt.subplot(411)
plt.xlim((0,1344))
plt.plot(y_data_flattened)
for i, row in enumerate(u):
    plt.subplot(4,1,i+2)
    plt.plot(row)
fig1.suptitle('Original input signal Decomposed modes')
plt.show()

fig2 = plt.figure()
plt.loglog(freqs[T//2:], abs(f_hat[T//2:]))
plt.xlim(np.array([1,T/2])*np.pi*2)
ax = plt.gca()
ax.grid(which='major', axis='both', linestyle='--')
fig2.suptitle('Input signal spectrum')


fig3 = plt.figure()
for k in range(K):
    plt.semilogx(2*np.pi/fs*omega[:,k], np.arange(1,omega.shape[0]+1), linestyles[k])
fig3.suptitle('Evolution of center frequencies omega')


fig4 = plt.figure()
plt.loglog(freqs[T//2:], abs(f_hat[T//2:]), 'k:')
plt.xlim(np.array([1, T//2])*np.pi*2)
for k in range(K):
    plt.loglog(freqs[T//2:], abs(u_hat[T//2:,k]), linestyles[k])
fig4.suptitle('Spectral decomposition')
plt.legend(['Original','1st component','2nd component','3rd component'])


fig4 = plt.figure()

for k in range(K):
    plt.subplot(3,1,k+1)
    plt.plot(t,u[k,:], linestyles[k])
    plt.plot(t, fsub[k+1], 'k:')

    plt.xlim((0,1))
    plt.title('Reconstructed mode %d'%(k+1))
plt.show()
#%%######################BP神经网络预测###############################
"""
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        # 初始化偏置
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def feedforward(self, inputs):
        # 输入到隐藏层
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        
        # 隐藏层到输出层
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)
        
        return final_outputs
    
    def train(self, inputs, targets, learning_rate):
        # 前向传播
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)
        
        # 计算输出层误差
        output_errors = targets - final_outputs
        output_delta = output_errors * self.sigmoid_derivative(final_outputs)
        
        # 计算隐藏层误差
        hidden_errors = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_errors * self.sigmoid_derivative(hidden_outputs)
        
        # 更新权重和偏置
        self.weights_hidden_output += np.dot(hidden_outputs.T, output_delta) * learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * learning_rate
        
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate


# 示例用法
# 创建一个具有2个输入，3个隐藏节点和1个输出的神经网络
nn = NeuralNetwork(2, 3, 1)

# 训练数据
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# 训练网络
epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
    nn.train(inputs, targets, learning_rate)

# 进行预测
for input in inputs:
    print(input, nn.feedforward(input))
"""




