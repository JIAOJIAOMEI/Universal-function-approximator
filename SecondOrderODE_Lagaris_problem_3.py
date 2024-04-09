# @Author  : Mei Jiaojiao
# @Time    : 2024/4/8 13:18
# @Software: PyCharm
# @File    : SecondOrderODE_Lagaris_problem_3.py


import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.autograd import grad


class DataSet(torch.utils.data.Dataset):

    def __init__(self, xRange, numSamples):
        self.InputData = torch.linspace(xRange[0], xRange[1], numSamples, requires_grad=True).view(-1, 1)

    def __len__(self):
        return len(self.InputData)

    def __getitem__(self, idx):
        return self.InputData[idx]


class Fitter(torch.nn.Module):

    def __init__(self, numHiddenNodes):
        super(Fitter, self).__init__()
        self.fully_connected_1 = torch.nn.Linear(in_features=1, out_features=numHiddenNodes)
        self.fully_connected_2 = torch.nn.Linear(in_features=numHiddenNodes, out_features=1)

    def forward(self, x, activation_Function):
        first_layer_output = self.fully_connected_1(x)
        first_layer_output = activation_Function(first_layer_output)
        y = self.fully_connected_2(first_layer_output)
        return y


def final_output(x, n_out):
    return x + (x ** 2 * n_out)


def first_derivative(x, n_out, dndx):
    return 1 + (2 * x * n_out) + (x ** 2 * dndx)


def second_derivative(x, n_out, dndx, d2ndx2):
    return 2 * n_out + (4 * x * dndx) + (x ** 2 * d2ndx2)


def Lagaris_problem_3(x, f_trial, df_trial, d2f_trial):
    LHS = d2f_trial + (1 / 5) * df_trial + f_trial
    RHS = -(1 / 5) * torch.exp(-x / 5) * torch.cos(x)
    return LHS - RHS


def solution_to_Lagaris_problem_3(x):
    return torch.exp(-x / 5) * torch.sin(x)


def train(neural_network, data_loader, loss_function, optimiser, num_Epochs, activationFn):
    cost_list = []
    neural_network.train(True)  # set module in training mode
    for epoch in range(num_Epochs):
        for batch in data_loader:
            n_out = neural_network.forward(batch, activationFn)
            dn_dx = grad(n_out, batch, torch.ones_like(n_out), retain_graph=True, create_graph=True)[0]
            dn_dx_2 = grad(dn_dx, batch, torch.ones_like(dn_dx), retain_graph=True)[0]

            # Get value of trial solution f(x)
            f_trial = final_output(batch, n_out)
            # Get df / dx
            df_trial = first_derivative(batch, n_out, dn_dx)
            # Get d^2f / dx^2
            d2f_trial = second_derivative(batch, n_out, dn_dx, dn_dx_2)
            # Get LHS of differential equation D(x) = 0
            diff_eq = Lagaris_problem_3(batch, f_trial, df_trial, d2f_trial)

            cost = loss_function(diff_eq, torch.zeros_like(diff_eq))
            # torch.zeros_like(x) creates a tensor the same shape as x, filled with 0's
            cost.backward()  # perform backpropagation
            optimiser.step()  # perform parameter optimisation
            optimiser.zero_grad()  # reset gradients to zero

        cost_list.append(cost.detach().numpy())  # store cost of each epoch
    neural_network.train(False)  # set module out of training mode
    return cost_list


x_range = [0, 10]
num_samples = 100
batch_size = 20
learning_rate = 1e-3
num_epochs = 1000
num_hidden_nodes = 20

network = Fitter(numHiddenNodes=num_hidden_nodes)

train_set = DataSet(xRange=x_range, numSamples=num_samples)
loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
lossFn = torch.nn.MSELoss()

optimiser_network = torch.optim.Adam(network.parameters(), lr=learning_rate)

activationFn_list = [torch.tanh, torch.relu, torch.sigmoid, torch.nn.functional.leaky_relu, torch.nn.functional.elu,
                     torch.nn.functional.hardswish]
num_epochs_list = [2000, 4000, 8000, 16000, 32000]

rows = len(activationFn_list)
cols = len(num_epochs_list)

num_samples_test = 2 * num_samples
fig, axs = plt.subplots(rows, cols, figsize=(50, 50))
for i, activationFn in enumerate(activationFn_list):
    for j, num_epochs in enumerate(num_epochs_list):
        cost_List = train(network, loader, lossFn, optimiser_network, num_epochs, activationFn)
        x = torch.linspace(x_range[0], x_range[1], num_samples_test).view(-1, 1)
        y_Exact = solution_to_Lagaris_problem_3(x).detach().numpy()
        y_Out = network.forward(x, activationFn).detach().numpy()
        x = x.detach().numpy()

        axs[i, j].plot(x, y_Exact, 'b.', label='Exact')
        axs[i, j].plot(x, final_output(x, y_Out), 'r.', label='Approx')
        axs[i, j].set_title(f'Activation Function: {activationFn.__name__}, Neural Network,{num_epochs} Epochs')
        axs[i, j].legend()

plt.savefig('SecondOrderODE_Lagaris_problem_3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('SecondOrderODE_Lagaris_problem_3.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
