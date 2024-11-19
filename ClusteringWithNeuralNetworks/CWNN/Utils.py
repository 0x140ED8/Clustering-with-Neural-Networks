import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import arff

def get_dataset_name(modelConfig):
    return modelConfig["select_dataset"]

def get_filepath(modelConfig):
    return modelConfig["data_filepath"] + modelConfig["dataset"][modelConfig["select_dataset"]][0]

def get_class_index(modelConfig):
    return modelConfig["dataset"][modelConfig["select_dataset"]][1]

def get_save_filepath(modelConfig):
    return modelConfig["save_filepath"] + modelConfig["select_dataset"] + "/" + modelConfig["select_dataset"] + "_" + modelConfig["model"]

def load_dataset_to_dataframe(filepath):
    file_extension = filepath.split('.')[-1].lower()

    if file_extension == 'csv':
        df = pd.read_csv(filepath, delimiter=';', skiprows=1, header=None)
    elif file_extension == 'xlsx':
        df = pd.read_excel(filepath)
    elif file_extension == 'xls':
        df = pd.read_excel(filepath)
    elif file_extension in ['data', 'txt']:
        df = pd.read_csv(filepath, header=None)
    elif file_extension == 'arff':
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))
        print(df.head())
    else:
        raise ValueError(f"不支持的文件类型: {file_extension}")

    return df


# Converts a partial encoding of a string in a data set to a numeric value
def convert(data):
    is_string = pd.Series(dtype=bool)

    for column in data:
        is_string[column] = data[column].apply(lambda x: isinstance(x, str)).any()

    for column_name, is_str in is_string.items():
        if is_str:
            unique_categories = data[column_name].unique()
            category_map = {category: code for code, category in enumerate(unique_categories)}
            data[column_name] = data[column_name].map(category_map)

# Sort by category group first, and then sort in ascending order within the group
def group_sort(sort_labels):
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

    return np.vstack(grouped_arrays)

def get_accuracys(result_array,lables,total):
    kmeans_inside_accuracys = []
    kmeans_outside_accuracys = []
    for i in range(0, 101):
        sum = 0
        train_rows = set()
        for j in lables:
            num = result_array[:, 0].tolist().count(j)
            train_rows.update(list(range(sum, math.ceil(sum + num * i / 100))))
            sum += num
        train_scope = result_array[list(train_rows)]
        test_rows = set(range(total)).difference(train_rows)
        test_scope = result_array[list(test_rows)]
        if train_scope.size > 0:
            train_accuracy = (train_scope[:, 2] == train_scope[:, 3]).sum() / len(train_scope)
        else:
            train_accuracy = None

        if test_scope.size > 0:
            test_accuracy = (test_scope[:, 2] == test_scope[:, 3]).sum() / len(test_scope)
        else:
            test_accuracy = None
        kmeans_inside_accuracys.append(train_accuracy)
        kmeans_outside_accuracys.append(test_accuracy)
    return kmeans_inside_accuracys,kmeans_outside_accuracys


def truncate_or_pad_list(lst, target_length, fill_value=None):
    return lst[:target_length] + [fill_value] * (target_length - len(lst))


def add_offset_to_list(data_list, offset):
    return [value + offset if value is not None else None for value in data_list]


def save_show(kmeans_inside_accuracys,kmeans_outside_accuracys,nn_ouside_accuracys,same_result_accuracys,modelConfig):

    max_length = max(len(kmeans_inside_accuracys), len(kmeans_outside_accuracys), len(nn_ouside_accuracys))

    kmeans_inside_accuracys = truncate_or_pad_list(kmeans_inside_accuracys, max_length)
    kmeans_outside_accuracys = truncate_or_pad_list(kmeans_outside_accuracys, max_length)
    nn_ouside_accuracys = truncate_or_pad_list(nn_ouside_accuracys, max_length)

    data = {
        'scope_percent' : list(range(0,101)),
        'cluster_inside_accuracys': kmeans_inside_accuracys,
        'cluster_outside_accuracys': kmeans_outside_accuracys,
        'nn_ouside_accuracys': nn_ouside_accuracys,
        'same_result_accuracys' : same_result_accuracys
    }

    dataset_name = get_dataset_name(modelConfig)
    save_filepath = get_save_filepath(modelConfig)
    if not os.path.exists(os.path.dirname(save_filepath)):
        os.makedirs(os.path.dirname(save_filepath))

    df = pd.DataFrame(data)
    df.to_csv(f"{save_filepath}.csv", index=False)


    fig, ax = plt.subplots(figsize=(10, 6))
    # offset = 0.01
    # nn_ouside_accuracys = add_offset_to_list(nn_ouside_accuracys,offset)
    ax.plot(list(range(0, 101)), kmeans_inside_accuracys, color="tab:blue", linestyle='-', linewidth=2,label="Kmeans Internal")
    ax.plot(list(range(0, 101)), kmeans_outside_accuracys, color="tab:green", linestyle='-.', linewidth=2,label="Kmeans External")
    ax.plot(list(range(0, 101)), nn_ouside_accuracys, color="tab:red", linestyle='--', linewidth=2,label=f"NN External")

    ax.set_xlabel("Internal division range(%)")    # Distance to cluster center
    ax.set_ylabel("Accuracy")
    ax.set_title(f"The accuracy of using CWNN({modelConfig['model']}) on the {dataset_name}")
    ax.grid(visible=True, linestyle='--', alpha=0.5)


    # ax.legend(loc='upper right', fontsize=12)
    plt.legend(loc='lower left', fontsize=12, ncol=3)
    ax.tick_params(axis='both', labelsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(f"{save_filepath}_accuracy.jpg",dpi=600)
    plt.show()