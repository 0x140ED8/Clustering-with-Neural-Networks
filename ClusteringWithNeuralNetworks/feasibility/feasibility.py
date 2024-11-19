import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from CWNN import get_filepath, load_dataset_to_dataframe, convert, KmeansModel


def get_max_length(data):

    return max(len(v) for v in data.values())


def pad_list(lst, target_length, fill_value=np.nan):
    """ Pads the list 'lst' to 'target_length' with 'fill_value'. """
    return lst + [fill_value] * (target_length - len(lst))

def pad_all(data):
    max_length = get_max_length(data)
    for key in data:
        data[key] = pad_list(data[key], max_length)




seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)


data = load_dataset_to_dataframe("../datasets/iris.data")

class_index = 4
data_index = list(range(data.shape[1]))     
if class_index != -1 and class_index != None:
    data_index.remove(class_index)
else:
    data_index.pop()

n_clusters = len(data.iloc[:, class_index].unique()) 
print("class:",n_clusters)
convert(data) 

print(data.head())

features = data.iloc[:, data_index].values  
labels = data.iloc[:, class_index].values 

features = features.astype('float64')
labels = labels.astype('float64')

X_tensor = torch.tensor(features, dtype=torch.float64)
labels_tensor = torch.tensor(labels, dtype=torch.float64)

cluster_model = KmeansModel(n_clusters = n_clusters,max_iter = 200)

cluster_model.fit(X_tensor)

cluster_assignments = cluster_model.labels_
unique_labels_ = torch.unique(cluster_model.labels_)
predicted_labels = torch.zeros_like(cluster_assignments)
for i in unique_labels_:
    cluster_points_mask = (cluster_assignments == i)
    predicted_labels[cluster_points_mask] = torch.mode(labels_tensor[cluster_points_mask]).values 

distances = torch.cdist(X_tensor, cluster_model.centers)
result = np.stack(
    (cluster_assignments.numpy(), torch.min(distances, dim=1).values.detach().numpy(), labels_tensor.numpy(),
     predicted_labels.numpy()), axis=1)

result = np.concatenate((result, features), axis=1)
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

plt.scatter(result[:50,6],result[:50,7],label='Setosa',marker='o',c='yellow')
plt.scatter(result[50:100,6],result[50:100,7],label='Versicolor',marker='o',c='green')
plt.scatter(result[100:,6],result[100:,7],label='Virginica',marker='o',c='blue')

plt.scatter(result[mask,6],result[mask,7],label='Prediction Error',marker='x',c='red')
plt.scatter(centers[:,2],centers[:,3],label='Center',marker='+',c='black',s=100)


plt.xlabel('petal length')		
plt.ylabel('petal width')	
plt.title("result of KMeans")
plt.legend()
# 保存为SVG格式
plt.savefig("feasibility.svg", format='svg')
plt.show()
