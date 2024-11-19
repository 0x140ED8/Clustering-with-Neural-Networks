import math
import torch
import numpy as np
from typing import Dict


from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

from CWNN import *
from CWNN import MeanShiftModel, load_dataset_to_dataframe, SimpleNN, KMeansModel_plusplus, convert, get_filepath


def cluster_run(modelConfig: Dict):
    seed = 88
    torch.manual_seed(seed)
    np.random.seed(seed)

    filepath = get_filepath(modelConfig)
    data = load_dataset_to_dataframe(filepath)

    class_index = get_class_index(modelConfig)
    if class_index is None:
        class_index = -1

    data_index = list(range(data.shape[1]))     # Feature column index
    if class_index != -1 and class_index != None:
        data_index.remove(class_index)
    else:
        data_index.pop()
    # print(data_index)

    n_clusters = len(data.iloc[:, class_index].unique())    # Class quantity
    print("class:",n_clusters)
    convert(data)   # Converts non-numeric data to numeric values

    print(data.head())

    # Extract feature and label columns
    features = data.iloc[:, data_index].values
    labels = data.iloc[:, class_index].values

    features = features.astype('float64')
    labels = labels.astype('float64')

    X_tensor = torch.tensor(features, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)

    # Choose the model
    if modelConfig["model"] == "kmeans":
        cluster_model = KmeansModel(n_clusters = n_clusters,max_iter = modelConfig["kmeans_epochs"])
    elif modelConfig["model"] == "kmeans++":
        cluster_model = KMeansModel_plusplus(n_clusters = n_clusters,max_iter = modelConfig["kmeans_epochs"])
    elif modelConfig["model"] == "meanshift":
        cluster_model = MeanShiftModel(max_iter = modelConfig["kmeans_epochs"])
    else:
        raise Exception('model:{}illegality'.format(modelConfig["model"]))

    cluster_model.fit(X_tensor)

    cluster_assignments = cluster_model.labels_
    unique_labels_ = torch.unique(cluster_model.labels_)
    predicted_labels = torch.zeros_like(cluster_assignments)
    for i in unique_labels_:
        cluster_points_mask = (cluster_assignments == i)
        # Use the category with the largest proportion as the prediction label
        predicted_labels[cluster_points_mask] = torch.mode(labels_tensor[cluster_points_mask]).values   # torch.mode众数

    distances = torch.cdist(X_tensor, cluster_model.centers)
    sort_labels = np.stack(
        (cluster_assignments.numpy(), torch.min(distances, dim=1).values.detach().numpy(), labels_tensor.numpy(),
         predicted_labels.numpy()), axis=1)

    sort_labels = np.concatenate((sort_labels, features), axis=1)
    # sort_labels [Owning cluster tag, distance from cluster center, actual tag, kmeans prediction tag, data, data, data......]

    # Sort by category group first, and then sort in ascending order within the group
    return group_sort(sort_labels),cluster_model






def train(modelConfig: Dict):
    result_array,cluster_model = cluster_run(modelConfig)
    # result_array    [Owning cluster tag, distance from cluster center, actual tag, kmeans prediction tag, data, data, data......]
    n_clusters = cluster_model.n_clusters
    total = len(result_array)  # Sample count
    feature = len(result_array[0][4:])  # Feature total
    print("n_clusters:",n_clusters)

    device = torch.device(modelConfig["device"])
    nn_ouside_accuracys = []
    same_result_accuracys = []

    unique_labels_ = torch.unique(cluster_model.labels_)
    # The internal and external accuracy rates of clustering in different ranges are obtained
    kmeans_inside_accuracys,kmeans_outside_accuracys = get_accuracys(result_array,unique_labels_,total)

    for scope in range(0, 101):
        if scope == 0 or scope == 1000: # If the number of test samples is 0, the test cannot be performed
            nn_ouside_accuracys.append(None)
            same_result_accuracys.append(None)
            continue

        sum = 0
        train_rows = set()  # Training set sample row index
        for j in unique_labels_:
            num = result_array[:, 0].tolist().count(j)
            train_rows.update(list(range(sum, math.ceil(sum + num * scope /100))))
            sum += num
        test_rows = set(range(total)).difference(train_rows)  # Test set sample row index
        train_columns = list(range(4, 4 + feature))  # Training set sample column index
        test_columns = list(range(4, 4 + feature))  # Test set sample column index
        train_columns.append(3)
        test_columns.append(2)

        train_data = result_array[list(train_rows)][:, train_columns]
        test_data = result_array[list(test_rows)][:, test_columns]
        X_train_tensor = torch.tensor(train_data[:, 0:-1], dtype=torch.float64).to(device)
        y_train_tensor = torch.tensor(train_data[:, -1], dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(test_data[:, 0:-1], dtype=torch.float64).to(device)
        y_test_tensor = torch.tensor(test_data[:, -1], dtype=torch.long).to(device)

        test_samples = len(y_test_tensor)
        if test_samples == 0:
            nn_ouside_accuracys.append(None)
            same_result_accuracys.append(None)
            continue

        test_kmeans_predicted = torch.Tensor(result_array[list(test_rows), 3]).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=modelConfig["batch_size"], shuffle=True)

        input_size = feature
        num_classes = n_clusters

        # Initialize the model, loss function, and optimizer
        model = SimpleNN(input_size, modelConfig["hidden_size1"], modelConfig["hidden_size2"],
                         modelConfig["hidden_size3"],
                         num_classes).double().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=modelConfig["learning_rate"])

        # Train the neural network model
        for epoch in range(modelConfig["nn_epochs"]):
            for i, (inputs, labels) in enumerate(train_loader):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        # Model evaluation
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            if test_samples != 0:
                nn_ouside_accuracy = (predicted == y_test_tensor).sum().item() / test_samples
                same_result_accuracy = (predicted == test_kmeans_predicted).sum().item() / test_samples
            else:
                nn_ouside_accuracy = None
                same_result_accuracy = None
        print(f"epoch:{scope}%:{nn_ouside_accuracy}")
        nn_ouside_accuracys.append(nn_ouside_accuracy)
        same_result_accuracys.append(same_result_accuracy)

    save_show(kmeans_inside_accuracys,kmeans_outside_accuracys,nn_ouside_accuracys,same_result_accuracys,modelConfig)