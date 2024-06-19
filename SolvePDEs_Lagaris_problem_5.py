# @Author  : Mei Jiaojiao
# @Time    : 2024/4/8 13:18
# @Software: PyCharm
# @File    : SolvePDEs_Lagaris_problem_5.py


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch.autograd import grad


class DataSet(torch.utils.data.Dataset):

    def __init__(self, xRange, yRange, numSamples):
        X = torch.linspace(xRange[0], xRange[1], numSamples, requires_grad=True)
        Y = torch.linspace(yRange[0], yRange[1], numSamples, requires_grad=True)
        grid = torch.meshgrid(X, Y)
        self.InputData = torch.cat([grid[0].reshape(-1, 1), grid[1].reshape(-1, 1)], dim=1)

    def __len__(self):
        return self.InputData.shape[0]

    def __getitem__(self, idx):
        return self.InputData[idx]


class PDESolver(torch.nn.Module):

    def __init__(self, numHiddenNodes):
        super(PDESolver, self).__init__()
        self.fully_connected_1 = torch.nn.Linear(in_features=2, out_features=numHiddenNodes)
        self.fully_connected_2 = torch.nn.Linear(in_features=numHiddenNodes, out_features=50)
        self.fully_connected_3 = torch.nn.Linear(in_features=50, out_features=1)

    def forward(self, x, activation_Function):
        first_layer_output = self.fully_connected_1(x)
        first_layer_output = activation_Function(first_layer_output)
        second_layer_output = self.fully_connected_2(first_layer_output)
        second_layer_output = activation_Function(second_layer_output)
        y = self.fully_connected_3(second_layer_output)
        return y


def final_output(x, y, n_out):
    e_inv = np.exp(-1)
    first_term = (1 - x) * (y ** 3) + x * (1 + (y ** 3)) * e_inv + (1 - y) * x * (torch.exp(-x) - e_inv) + y * (
            (1 + x) * torch.exp(-x) - (1 - x + (2 * x * e_inv)))
    return first_term + x * (1 - x) * y * (1 - y) * n_out


def solution(x, y):
    return torch.exp(-x) * (x + y ** 3)


def first_partial_derivative_x(x, y, n_out, dndx):
    return (-torch.exp(-x) * (x + y - 1) - y ** 3 + (y ** 2 + 3) * y * np.exp(-1) + y
            + y * (1 - y) * ((1 - 2 * x) * n_out + x * (1 - x) * dndx))


def second_partial_derivative_x(x, y, n_out, dndx, d2ndx2):
    return (torch.exp(-x) * (x + y - 2) + y * (1 - y) * ((-2 * n_out) + 2 * (1 - 2 * x) * dndx) + x * (1 - x) * d2ndx2)


def first_partial_derivative_y(x, y, n_out, dndy):
    return (3 * x * (y ** 2 + 1) * np.exp(-1) - (x - 1) * (3 * (y ** 2) - 1) + torch.exp(-x)
            + x * (1 - x) * ((1 - 2 * y) * n_out + y * (1 - y) * dndy))


def second_partial_derivative_y(x, y, n_out, dndy, d2ndy2):
    return (np.exp(-1) * 6 * y * (-np.exp(1) * x + x + np.exp(1))
            + x * (1 - x) * ((-2 * n_out) + 2 * (1 - 2 * y) * dndy) + y * (1 - y) * d2ndy2)


def my_loss(x, y, y_2nd_derivative, x_2nd_derivative):
    RHS = torch.exp(-x) * (x - 2 + y ** 3 + 6 * y)
    return x_2nd_derivative + y_2nd_derivative - RHS


def train(neural_network, data_loader, loss_function, optimiser, num_Epochs, activationFn):
    cost_list = []
    neural_network.train(True)  # set module in training mode
    for epoch in range(num_Epochs):
        for batch in data_loader:
            n_out = neural_network.forward(batch, activationFn)
            dn_out = grad(n_out, batch, torch.ones_like(n_out), retain_graph=True, create_graph=True)[0]
            dn2_out = grad(dn_out, batch, torch.ones_like(dn_out), retain_graph=True, create_graph=True)[0]

            # get the first partial derivative of a trial solution
            dndx, dndy = torch.split(dn_out, split_size_or_sections=1, dim=1)
            # get second partial derivative of trial solution
            d2ndx2, d2ndy2 = torch.split(dn2_out, split_size_or_sections=1, dim=1)

            x, y = torch.split(batch, split_size_or_sections=1, dim=1)

            x_2nd_derivative = second_partial_derivative_x(x, y, n_out, dndx, d2ndx2)
            y_2nd_derivative = second_partial_derivative_y(x, y, n_out, dndy, d2ndy2)

            loss = my_loss(x, y, y_2nd_derivative, x_2nd_derivative)

            cost = loss_function(loss, torch.zeros_like(loss))

            optimiser.zero_grad()
            cost.backward()
            optimiser.step()

        cost_list.append(cost.detach().numpy())
    neural_network.train(False)
    return cost_list


x_range = [0, 1]
y_range = [0, 1]

network = PDESolver(numHiddenNodes=20)
batch_size = 20

lossFn = torch.nn.MSELoss()
optimiser_network = torch.optim.Adam(network.parameters(), lr=1e-3)
activationFn = torch.nn.Tanh()

num_samples_list = [5, 20, 40]
num_epochs_list = [100, 500, 1000]

rows = len(num_samples_list)
cols = len(num_epochs_list)

num_samples_test = np.random.randint(5, 10)
plt.figure(figsize=(20, 20))
for i, num_samples in enumerate(num_samples_list):
    for j, num_epochs in enumerate(num_epochs_list):
        train_set = DataSet(xRange=x_range, yRange=y_range, numSamples=num_samples)
        loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        cost_List = train(network, loader, lossFn, optimiser_network, num_epochs, activationFn)
        x = torch.linspace(x_range[0], x_range[1], num_samples_test, requires_grad=True)
        y = torch.linspace(y_range[0], y_range[1], num_samples_test, requires_grad=True)
        X, Y = torch.meshgrid(x, y)
        Input = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
        n_out = network.forward(Input, activationFn)
        final_out = final_output(X.reshape(-1, 1), Y.reshape(-1, 1), n_out)
        final_out = final_out.reshape(num_samples_test, num_samples_test).detach().numpy()
        y_Exact = solution(X, Y).detach().numpy()
        # residual error
        residual_error = np.sqrt((final_out - y_Exact) ** 2).mean()
        ax = plt.subplot(rows, cols, i * cols + j + 1, projection='3d')
        ax.plot_surface(X.detach().numpy(), Y.detach().numpy(), final_out, cmap='viridis', label='Predicted')
        ax.scatter(X.detach().numpy(), Y.detach().numpy(), y_Exact, c='r', label='Exact Solution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title('Epochs: ' + str(num_epochs) + ', Residual Error: ' + str(residual_error) + ", Samples: " + str(
            num_samples))

plt.savefig('SolvePDEs_Lagaris_problem_5.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('SolvePDEs_Lagaris_problem_5.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
