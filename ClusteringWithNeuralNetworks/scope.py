import torch
from sklearn.metrics import accuracy_score
import math
import numpy as np
import matplotlib.pyplot as plt

from CWNN import load_dataset_to_dataframe, convert


### It is used to test the internal and external accuracy of clustering algorithm in different scope values


filepath = './datasets/iris.data'
select_scope = 0.1
seed = 88
torch.manual_seed(seed)
np.random.seed(seed)


data = load_dataset_to_dataframe(filepath)

n_clusters = len(data.iloc[:, -1].unique())
print(len(data))

convert(data)

features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values

features = features.astype('float64')
labels = labels.astype('float64')

X_tensor = torch.tensor(features, dtype=torch.float64)
labels_tensor = torch.tensor(labels, dtype=torch.float64)


total = len(X_tensor)
feature = len(X_tensor[0])

device = torch.device("cpu")
X_tensor = X_tensor.to(device)


class KMeansModel(torch.nn.Module):
    def __init__(self, n_clusters):
        super(KMeansModel, self).__init__()
        self.centers = torch.nn.Parameter(X_tensor[np.random.choice(total, n_clusters, replace=False)])

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        return distances


kmeans_model = KMeansModel(n_clusters).to(device)

optimizer = torch.optim.Adam(kmeans_model.parameters(), lr=0.01)


tolerance = 1e-4
num_epochs = 100
for epoch in range(num_epochs):
    distances = kmeans_model(X_tensor)
    _, cluster_assignments = torch.min(distances, dim=-1)

    previous_centers = kmeans_model.centers.data.clone()

    for i in range(n_clusters):
        cluster_points = X_tensor[cluster_assignments == i]
        if len(cluster_points) > 0:
            kmeans_model.centers.data[i] = torch.mean(cluster_points, dim=0).cpu()

    center_shift = torch.sum(torch.sqrt(torch.sum((kmeans_model.centers.data - previous_centers) ** 2, dim=1)))

    if center_shift < tolerance:
        print("K-Means converges on the {}th iteration".format(epoch + 1))
        break

cluster_assignments = torch.min(distances, dim=-1).indices


predicted_labels = torch.zeros_like(cluster_assignments)
for i in range(n_clusters):
    cluster_points_mask = (cluster_assignments == i)
    predicted_labels[cluster_points_mask] = torch.mode(torch.tensor(labels[cluster_points_mask])).values.item()

sort_labels = np.stack((cluster_assignments.numpy(), torch.min(distances, dim=1).values.detach().numpy(), labels,
                        predicted_labels.numpy()), axis=1)
sort_labels = np.concatenate((sort_labels, X_tensor), axis=1)


print(sort_labels.shape)

sorted_array = sort_labels[sort_labels[:, 0].argsort()]

grouped_data = {}
for row in sorted_array:
    key = row[0]
    if key not in grouped_data:
        grouped_data[key] = []
    grouped_data[key].append(row)

for key, group in grouped_data.items():
    grouped_data[key] = np.array(sorted(group, key=lambda x: x[1]))

grouped_arrays = list(grouped_data.values())

result_array = np.vstack(grouped_arrays)


train_accuracy_list = []
test_accuracy_list = []

for i in range(1, 101):
    sum = 0
    train_rows = set()
    for j in range(n_clusters):
        num = result_array[:, 0].tolist().count(j)
        train_rows.update(list(range(sum, math.ceil(sum + num * i / 100))))
        sum += num
    train_scope = result_array[list(train_rows)]
    test_rows = set(range(total)).difference(train_rows)
    test_scope = result_array[list(test_rows)]

    if train_scope.size > 0:
        train_accuracy = accuracy_score(train_scope[1:, 2], train_scope[1:, 3])
    else:
        train_accuracy = None

    if test_scope.size > 0:
        test_accuracy = accuracy_score(test_scope[1:, 2], test_scope[1:, 3])
    else:
        test_accuracy = None

    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)


data_part = list(range(4, 4 + feature))
data_part.append(2)
train = result_array[list(train_rows)][:, data_part]
test = result_array[list(test_rows)][:, data_part]

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.plot(list(range(1, 101)), train_accuracy_list, "g", marker='D', markersize=2, label="Internal accuracy")
plt.plot(list(range(1, 101)), test_accuracy_list, "r", marker='D', markersize=2, label="External accuracy")
plt.xlabel("scope")
plt.ylabel("Accuracy rate")
plt.title("kmeans different scope accuracy")
plt.legend(loc="lower right")

plt.savefig(f"{filepath}different scope accuracy.jpg")
plt.show()