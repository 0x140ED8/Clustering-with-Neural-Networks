from CWNN.Train import *

def main():
    modelConfig = {
        "model" : "kmeans",# kmeans,kmeans++,meanshift
        "save_filepath" : "./save/",
        "data_filepath" : "./datasets/",  # The file path to the data set
        "select_dataset" : "Iris",  # Selective data set
        "dataset" : {
            # name :        [filename, class_index]
            "Iris" :        ["iris.data",-1], # https://archive.ics.uci.edu/dataset/53/iris
            "Abalone" :     ["abalone.data",0], # https://archive.ics.uci.edu/dataset/1/abalone
            "Dry Bean" :    ["Dry_Bean_Dataset.xlsx",-1], # https://archive.ics.uci.edu/dataset/602/dry+bean+dataset  # The result is abnormal and cannot be used
            "Rice" :        ["Rice_Cammeo_Osmancik.arff",-1], # https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
            "Adult" :       ["adult.data",-1], # https://archive.ics.uci.edu/dataset/2/adult
            "Car Evaluation" :  ["car.data",-1], # https://archive.ics.uci.edu/dataset/19/car+evaluation
            "Breast Cancer" :   ["breast-cancer.data",0], # https://archive.ics.uci.edu/dataset/14/breast+cancer
            "Breast Cancer Wisconsin": ["wdbc.data", 1], # https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
            "Predict Students' Dropout and Academic Success": ["data.csv", -1], # https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
            "Bank Marketing" : ["bank.csv",-1],   # https://archive.ics.uci.edu/dataset/222/bank+marketing
            "Mushroom" : ["agaricus-lepiota.data",0]    # https://archive.ics.uci.edu/dataset/73/mushroom
        },

        "device" : "cuda:0",    # cuda:0 or cpu
        "kmeans_epochs" : 100,  # K-means maximum number of iterations
        "nn_epochs": 1000,  # Number of neural network iterations
        "hidden_size1" : 8,     # Set the number of hidden neurons in the first layer of the hyperparameter neural network
        "hidden_size2" : 16,
        "hidden_size3" : 32,
        "batch_size" : 512,
        "learning_rate" : 0.001,
        }

    train(modelConfig)

# This project has realized a large number of automated operations on data sets, but some special data sets are difficult to operate
# Attention!When using MeanShift, the quantile coefficient in the estimate_bandwidth method in the MeanShiftModel class should be adjusted so that the number of clusters is close to the total number of classes (at least less than 100).
if __name__ == '__main__':
    main()
