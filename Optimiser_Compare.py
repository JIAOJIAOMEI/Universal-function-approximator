# @Author  : Mei Jiaojiao
# @Time    : 2024/4/8 10:18
# @Software: PyCharm
# @File    : FunctionApproximation.py


import matplotlib.pyplot as plt
import torch
from jax import random

rand_key = random.PRNGKey(0)


class DataSet(torch.utils.data.Dataset):

    def __init__(self, my_underlying_function, xRange, numSamples):
        self.InputData = torch.linspace(xRange[0], xRange[1], numSamples).view(-1, 1)
        self.OutputData = my_underlying_function(self.InputData).view(-1, 1)

    def __len__(self):
        return len(self.InputData)

    def __getitem__(self, idx):
        return self.InputData[idx], self.OutputData[idx]


def fourier_feature_mappping_function(x):
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class Fitter(torch.nn.Module):

    def __init__(self, numHiddenNodes):
        super(Fitter, self).__init__()
        self.fully_connected_1 = torch.nn.Linear(in_features=2, out_features=numHiddenNodes)
        self.fully_connected_2 = torch.nn.Linear(in_features=numHiddenNodes, out_features=1)

    def forward(self, x, activation_Function):
        first_layer_output = self.fully_connected_1(fourier_feature_mappping_function(x))
        first_layer_output = activation_Function(first_layer_output)
        y = self.fully_connected_2(first_layer_output)
        return y


def train(neural_network, data_loader, loss_function, optimiser, num_Epochs, activationFn):
    cost_List = []
    neural_network.train(True)  # set module in training mode
    for epoch in range(num_Epochs):
        for batch in data_loader:
            x, y = batch[0], batch[1]  # x and y=f(x) values
            y_Out = neural_network.forward(x, activationFn)  # forward pass
            cost = loss_function(y_Out, y)  # calculate cost value
            cost.backward()  # back propagation, calculate gradients
            optimiser.step()  # perform parameter optimisation
            optimiser.zero_grad()  # reset gradients to zero
        cost_List.append(cost.item())  # store cost of each epoch
    neural_network.train(False)  # set module out of training mode
    return cost_List


my_underlying_function = torch.cos
num_hidden_nodes = 50
x_range = [-10, 10]
num_samples = 100
batch_size = 30
learning_rate = 1e-4
numEpochs = 30

network = Fitter(numHiddenNodes=num_hidden_nodes)
trainSet = DataSet(my_underlying_function, xRange=x_range, numSamples=num_samples)
loader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batch_size)
lossFn = torch.nn.MSELoss()  # mean-squared error loss
optimiser = torch.optim.SGD(network.parameters(), lr=learning_rate)

activationFn_list = [torch.tanh, torch.relu, torch.sigmoid, torch.nn.functional.leaky_relu, torch.nn.functional.elu,
                     torch.nn.functional.hardswish]
optimiser_list = [torch.optim.SGD, torch.optim.Adam, torch.optim.Adagrad, torch.optim.Adadelta, torch.optim.AdamW,
                    torch.optim.Adamax, torch.optim.RMSprop]
optimiser_name_list = ['SGD', 'Adam', 'Ada-grad', 'Ada-delta', 'AdamW', 'Adam', 'RMSprop']


rows = len(activationFn_list)
cols = len(optimiser_list)

num_samples_test = 2*num_samples
fig, axs = plt.subplots(rows, cols, figsize=(80, 50))
for i, activationFn in enumerate(activationFn_list):
    for j, optimiser in enumerate(optimiser_list):
        optimiser = optimiser(network.parameters(), lr=learning_rate)
        costList = train(network, loader, lossFn, optimiser, numEpochs, activationFn)
        x = torch.linspace(x_range[0], x_range[1], num_samples_test).view(-1, 1)
        y_Exact = my_underlying_function(x)
        y_Out = network.forward(x, activationFn).detach().numpy()
        x = x.detach().numpy()
        axs[i, j].plot(x, y_Exact, 'b.', label='Exact Solution y = cos(x)')
        axs[i, j].plot(x, y_Out, 'r.', label='Neural Network Output')
        axs[i, j].set_title(f'{activationFn.__name__} Activation Function, {optimiser_name_list[j]} Optimiser, {numEpochs} Epochs', fontsize=16)
        axs[i, j].legend(loc='upper left', fontsize=13)
plt.savefig('Optimiser_Compare.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('Optimiser_Compare.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
