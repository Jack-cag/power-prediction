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
