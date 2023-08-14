import os
import shutil
from sklearn.model_selection import train_test_split
import random
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
# 定义要汇总的文件夹列表和目标文件夹
folder_list = ['E:\\t_map\\extract_data\\s7test\\100\\points',
               'E:\\t_map\\extract_data\\s7test\\200\\points',
               'E:\\t_map\\extract_data\\s7test\\300\\points',
               'E:\\t_map\\extract_data\\s7test\\400\\points',
               'E:\\t_map\\extract_data\\s7test\\500\\points']

target_folder = 'E:\\t_map\\extract_data\\s7test\\1'

# 创建目标文件夹
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

# 遍历文件夹列表，将csv文件复制到目标文件夹，并按照指定方式命名
for folder in folder_list:
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            new_file_name = os.path.basename(os.path.dirname(folder)) + '_' + file
            shutil.copy(os.path.join(folder, file), os.path.join(target_folder, new_file_name))
# #划分数据集
# train_ratio = 0.7
# test_ratio = 0.2
# val_ratio = 0.1

# 创建训练集、测试集、验证集文件夹
train_folder = os.path.join(target_folder, 'train')
test_folder = os.path.join(target_folder, 'test')
val_folder = os.path.join(target_folder, 'val')
if not os.path.exists(train_folder):
    os.mkdir(train_folder)
if not os.path.exists(test_folder):
    os.mkdir(test_folder)
if not os.path.exists(val_folder):
    os.mkdir(val_folder)
    
csv_files = [os.path.join(target_folder, file) for file in os.listdir(target_folder) if file.endswith('.csv')]

# # 随机划分数据集
# train_files, test_val_files = train_test_split(csv_files, train_size=train_ratio, random_state=1)
# test_files, val_files = train_test_split(test_val_files, train_size=test_ratio/(test_ratio+val_ratio), random_state=1)
data = []
# 对每个 CSV 文件进行数据集划分
for csv_file in csv_files:
    # 加载 CSV 文件并获取分层依据
    df = pd.read_csv(csv_file)
    t = df.iloc[:, 2].max()
    data.append({'filepath': csv_file, 't': t})
    
# 按照标签值进行排序
data_sorted = sorted(data, key=lambda x: x['t'])

#排序后根据index每10个中7个放train，2个放test，1个放val
for i, item in enumerate(data_sorted):
    filepath = item['filepath']

    # if i % 20 < 8 :
    #     folder = train_folder
    # elif i % 20 < 9 :
    #     folder = test_folder
    # elif i % 20 < 10 :
    #     folder = val_folder
    # elif i % 20 < 19 :
    #     folder = train_folder    
    # else:
    #     folder = test_folder

    # 8:2
    if i % 5 < 4 :
        folder = train_folder
    else :
        folder = test_folder
    shutil.copy(filepath, os.path.join(folder, os.path.basename(filepath)))
print("划分完成")    
# # 将文件复制到相应的文件夹中
# for index in train_indices:
#     src_file = data[index]['filepath']
#     shutil.copy(src_file, os.path.join(train_folder, os.path.basename(src_file)))
# for index in test_indices:
#     src_file = data[index]['filepath']
#     shutil.copy(src_file, os.path.join(test_folder, os.path.basename(src_file)))
# for index in val_indices:
#     src_file = data[index]['filepath']
#     shutil.copy(src_file, os.path.join(val_folder, os.path.basename(src_file)))  