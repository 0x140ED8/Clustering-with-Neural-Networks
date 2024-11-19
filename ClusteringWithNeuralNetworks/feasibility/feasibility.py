import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from CWNN import get_filepath, load_dataset_to_dataframe, convert, KmeansModel


def get_max_length(data):
    # 计算最长列的长度
    return max(len(v) for v in data.values())

# 定义填充函数
def pad_list(lst, target_length, fill_value=np.nan):
    """ Pads the list 'lst' to 'target_length' with 'fill_value'. """
    return lst + [fill_value] * (target_length - len(lst))

def pad_all(data):
    max_length = get_max_length(data)
    # 为每一列填充缺失值到相同长度
    for key in data:
        data[key] = pad_list(data[key], max_length)




seed = 42  # 设置随机种子
torch.manual_seed(seed)
np.random.seed(seed)

# 加载数据集,读取Excel文件
data = load_dataset_to_dataframe("../datasets/iris.data")

class_index = 4
data_index = list(range(data.shape[1]))     # 特征列索引
if class_index != -1 and class_index != None:
    data_index.remove(class_index)
else:
    data_index.pop()

n_clusters = len(data.iloc[:, class_index].unique())    # 类别数量
print("class:",n_clusters)
convert(data)   # 将非数值数据转换为数值

print(data.head())

# 提取特征和标签列
features = data.iloc[:, data_index].values  # 使用 data_index 作为特征列索引
labels = data.iloc[:, class_index].values  # 使用 class_index 作为标签列索引

# 将特征和标签数组转换为 float64 类型
features = features.astype('float64')
labels = labels.astype('float64')

# 将特征和标签转换为PyTorch张量
X_tensor = torch.tensor(features, dtype=torch.float64)
labels_tensor = torch.tensor(labels, dtype=torch.float64)

cluster_model = KmeansModel(n_clusters = n_clusters,max_iter = 200)

cluster_model.fit(X_tensor)

cluster_assignments = cluster_model.labels_
unique_labels_ = torch.unique(cluster_model.labels_)
predicted_labels = torch.zeros_like(cluster_assignments)
for i in unique_labels_:
    cluster_points_mask = (cluster_assignments == i)
    # 使用占比最多的类别作为预测标签
    predicted_labels[cluster_points_mask] = torch.mode(labels_tensor[cluster_points_mask]).values   # torch.mode众数

distances = torch.cdist(X_tensor, cluster_model.centers)
result = np.stack(
    (cluster_assignments.numpy(), torch.min(distances, dim=1).values.detach().numpy(), labels_tensor.numpy(),
     predicted_labels.numpy()), axis=1)

result = np.concatenate((result, features), axis=1)
# sort_labels [所属簇，距离簇心距离，实际标签，kmeans预测标签，数据，数据，数据......]
mask = result[:,2] != result[:,3]
centers = cluster_model.centers.detach().numpy()

length = len(result)

data = {
        'petal length': result[:,6].tolist(),
        'petal width': result[:,7].tolist(),
        'class' : result[:,2].tolist(),
        'predict class' : result[:,3].tolist(),
        'Error length': result[mask,6].tolist(),
        'Error width': result[mask,7].tolist(),
        'Centers length' : centers[:,2].tolist(),
        'Centers width' : centers[:,3].tolist()
    }
pad_all(data)
df = pd.DataFrame(data)
df.to_csv("feasibility.csv", index=False)

# 原始数据的特征散点图
plt.scatter(result[:50,6],result[:50,7],label='Setosa',marker='o',c='yellow')
plt.scatter(result[50:100,6],result[50:100,7],label='Versicolor',marker='o',c='green')
plt.scatter(result[100:,6],result[100:,7],label='Virginica',marker='o',c='blue')

plt.scatter(result[mask,6],result[mask,7],label='Prediction Error',marker='x',c='red')
plt.scatter(centers[:,2],centers[:,3],label='Center',marker='+',c='black',s=100)


plt.xlabel('petal length')			# 花瓣长
plt.ylabel('petal width')			# 花瓣宽
plt.title("result of KMeans")
plt.legend()
# 保存为SVG格式
plt.savefig("feasibility.svg", format='svg')
plt.show()
