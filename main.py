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





