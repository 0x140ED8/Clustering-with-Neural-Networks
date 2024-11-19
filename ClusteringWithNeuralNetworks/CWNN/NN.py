import torch.nn as nn

# Define a 3-layer neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size3, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


# Define a layer neural network model
# class SimpleNN(nn.Module):
#     def __init__(self, input_size, hidden_size1, num_classes):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size1, num_classes)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         return out