# @Author  : Mei Jiaojiao
# @Time    : 2024/4/8 13:18
# @Software: PyCharm
# @File    : SolvePDEs_Lagaris_problem_6.py


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch.autograd import grad


class LinearDataSet(torch.utils.data.Dataset):

    def __init__(self, xRange, yRange, numSamples):
        X = torch.linspace(xRange[0], xRange[1], numSamples, requires_grad=True)
        Y = torch.linspace(yRange[0], yRange[1], numSamples, requires_grad=True)
        grid = torch.meshgrid(X, Y)
        self.InputData = torch.cat([grid[0].reshape(-1, 1), grid[1].reshape(-1, 1)], dim=1)

    def __len__(self):
        return self.InputData.shape[0]

    def __getitem__(self, idx):
        return self.InputData[idx]


class UniformDataSet(torch.utils.data.Dataset):

    def __init__(self, xRange, yRange, numSamples):
        X = torch.distributions.Uniform(xRange[0], xRange[1]).sample((int(numSamples ** 2), 1))
        X.requires_grad = True
        Y = torch.distributions.Uniform(yRange[0], yRange[1]).sample((int(numSamples ** 2), 1))
        Y.requires_grad = True
        self.InputData = torch.cat([X, Y], dim=1)

    def __len__(self):
        return self.InputData.shape[0]

    def __getitem__(self, idx):
        return self.InputData[idx]


class NormalDataSet(torch.utils.data.Dataset):

    def __init__(self, xRange, yRange, numSamples):
        X = torch.distributions.Normal(0, 1).sample((int(numSamples ** 2), 1))
        X = (X - X.min()) / (X.max() - X.min()) * (xRange[1] - xRange[0]) + xRange[0]
        X.requires_grad = True
        Y = torch.distributions.Normal(0, 1).sample((int(numSamples ** 2), 1))
        Y = (Y - Y.min()) / (Y.max() - Y.min()) * (yRange[1] - yRange[0]) + yRange[0]
        Y.requires_grad = True
        self.InputData = torch.cat([X, Y], dim=1)

    def __len__(self):
        return self.InputData.shape[0]

    def __getitem__(self, idx):
        return self.InputData[idx]


class PDESolver(torch.nn.Module):

    def __init__(self, numHiddenNodes):
        super(PDESolver, self).__init__()
        self.fully_connected_1 = torch.nn.Linear(in_features=2, out_features=numHiddenNodes[0])
        self.fully_connected_2 = torch.nn.Linear(in_features=numHiddenNodes[0], out_features=numHiddenNodes[1])
        self.fully_connected_3 = torch.nn.Linear(in_features=numHiddenNodes[1], out_features=1)

    def forward(self, x, activation_Function):
        first_layer_output = self.fully_connected_1(x)
        first_layer_output = activation_Function(first_layer_output)
        second_layer_output = self.fully_connected_2(first_layer_output)
        second_layer_output = activation_Function(second_layer_output)
        y = self.fully_connected_3(second_layer_output)
        return y


def final_output(x, y, n_outXY, n_outX1, n_outX1_y):
    first_term = y ** 2 * torch.sin(np.pi * x)
    return first_term + x * (1 - x) * y * (n_outXY - n_outX1 - n_outX1_y)


def solution(x, y):
    return (y ** 2) * torch.sin(np.pi * x)


def first_partial_derivative_x(x, y, n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy):
    return (y ** 2 * np.pi * torch.cos(np.pi * x) + y * ((1 - 2 * x) * (n_outXY - n_outX1 - n_outX1_y)
                                                         + x * (1 - x) * (n_outXY_x - n_outX1_x - n_outX1_xy)))


def second_partial_derivative_x(x, y, n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy, n_outXY_xx,
                                n_outX1_xx, n_outX1_xxy):
    return -y ** 2 * np.pi ** 2 * torch.sin(np.pi * x) + y * (
            (-2) * (n_outXY - n_outX1 - n_outX1_y) + 2 * (1 - 2 * x) * (n_outXY_x - n_outX1_x - n_outX1_xy) + x * (
            1 - x) * (n_outXY_xx - n_outX1_xx - n_outX1_xxy))


def first_partial_derivative_y(x, y, n_outXY, n_outX1, n_outX1_y, n_outXY_y):
    return 2 * y * torch.sin(np.pi * x) + x * (1 - x) * ((n_outXY - n_outX1 - n_outX1_y) + (y * n_outXY_y))


def second_partial_derivative_y(x, y, n_outXY_y, n_outXY_yy):
    return 2 * torch.sin(np.pi * x) + x * (1 - x) * (2 * n_outXY_y + y * n_outXY_yy)


def my_loss(x, y, trialFunc, trial_dy, trial_dx2, trial_dy2):
    RHS = torch.sin(np.pi * x) * (2 - np.pi ** 2 * y ** 2 + 2 * y ** 3 * torch.sin(np.pi * x))
    return trial_dx2 + trial_dy2 + trialFunc * trial_dy - RHS


def train(neural_network, data_loader, loss_function, optimiser, num_Epochs, activationFn):
    cost_list = []
    neural_network.train(True)  # set module in training mode
    for epoch in range(num_Epochs):
        for batch in data_loader:
            x, y = torch.split(batch, split_size_or_sections=1, dim=1)
            x1 = torch.cat((x, torch.ones_like(y)), dim=1)

            n_outXY = network(batch, activationFn)  # Neural network output at (x,y)
            n_outX1 = network(x1, activationFn)  # Neural network output at (x,1)

            # Get all required derivatives of n(x,y)
            grad_n_outXY = grad(n_outXY, batch, torch.ones_like(n_outXY), retain_graph=True, create_graph=True)[0]
            n_outXY_x, n_outXY_y = torch.split(grad_n_outXY, 1, dim=1)  # n_x , n_y

            grad_grad_n_outXY = \
                grad(grad_n_outXY, batch, torch.ones_like(grad_n_outXY), retain_graph=True, create_graph=True)[0]
            n_outXY_xx, n_outXY_yy = torch.split(grad_grad_n_outXY, 1, dim=1)  # n_xx , n_yy

            # Get all required derivatives of n(x,1):
            grad_n_outX1 = grad(n_outX1, x1, torch.ones_like(n_outX1), retain_graph=True, create_graph=True)[0]
            n_outX1_x, n_outX1_y = torch.split(grad_n_outX1, 1, dim=1)  # n_x |(y=1) , n_y |(y=1)

            grad_n_outX1_x = grad(n_outX1_x, x1, torch.ones_like(n_outX1_x), retain_graph=True, create_graph=True)[0]
            n_outX1_xx, n_outX1_xy = torch.split(grad_n_outX1_x, 1, dim=1)  # n_xx |(y=1), n_xy |(y=1)

            grad_n_outX1_xy = grad(n_outX1_xy, x1, torch.ones_like(n_outX1_xy), retain_graph=True, create_graph=True)[0]
            n_outX1_xxy, _ = torch.split(grad_n_outX1_xy, 1, dim=1)  # n_xxy |(y=1)

            # Get a trial solution
            trialFunc = final_output(x, y, n_outXY, n_outX1, n_outX1_y)  # f(x,y)
            trial_dy = first_partial_derivative_y(x, y, n_outXY, n_outX1, n_outX1_y, n_outXY_y)  # f_y
            trial_dx2 = second_partial_derivative_x(x, y, n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy,
                                                    n_outXY_xx, n_outX1_xx, n_outX1_xxy)  # f_xx
            trial_dy2 = second_partial_derivative_y(x, y, n_outXY_y, n_outXY_yy)  # f_yy

            loss = my_loss(x, y, trialFunc, trial_dy, trial_dx2, trial_dy2)

            cost = loss_function(loss, torch.zeros_like(loss))

            optimiser.zero_grad()
            cost.backward()
            optimiser.step()
            optimiser.zero_grad()

        cost_list.append(cost.detach().numpy())
    neural_network.train(False)
    return cost_list


x_range = [0, 1]
y_range = [0, 1]

network = PDESolver(numHiddenNodes=[20, 30])
batch_size = 20
num_epochs = 100

lossFn = torch.nn.MSELoss()
optimiser_network = torch.optim.Adam(network.parameters(), lr=1e-4)
activationFn = torch.nn.Tanh()

num_samples_list = [20, 40, 60]
samplingMethod = [LinearDataSet, UniformDataSet, NormalDataSet]
str_list = ['Linear', 'Uniform', 'Normal']

rows = len(num_samples_list)
cols = len(samplingMethod)

num_samples_test = np.random.randint(10, 12)
plt.figure(figsize=(30, 30))
for i, num_samples in enumerate(num_samples_list):
    for j, DataSet in enumerate(samplingMethod):
        train_set = DataSet(xRange=x_range, yRange=y_range, numSamples=num_samples)
        loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        cost_List = train(network, loader, lossFn, optimiser_network, num_epochs, activationFn)
        x = torch.linspace(x_range[0], x_range[1], num_samples_test, requires_grad=True)
        y = torch.linspace(y_range[0], y_range[1], num_samples_test, requires_grad=True)
        X, Y = torch.meshgrid(x, y)
        Input = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
        x_split, y_split = torch.split(Input, 1, dim=1)
        n_outXY = network.forward(Input, activationFn)
        x1 = torch.cat((x_split, torch.ones_like(y_split)), dim=1)
        n_outX1 = network.forward(x1, activationFn)

        grad_n_outX1 = grad(n_outX1, x1, torch.ones_like(n_outX1), retain_graph=True, create_graph=True)[0]
        n_outX1_x, n_outX1_y = torch.split(grad_n_outX1, 1, dim=1)

        final_out = final_output(x_split, y_split, n_outXY, n_outX1, n_outX1_y).detach().numpy()
        y_Exact = solution(x_split, y_split).detach().numpy()
        # residual error
        residual_error = np.sqrt((final_out - y_Exact) ** 2).mean()
        final_out = final_out.reshape(num_samples_test, num_samples_test)
        ax = plt.subplot(rows, cols, i * cols + j + 1, projection='3d')
        ax.plot_surface(X.detach().numpy(), Y.detach().numpy(), final_out, cmap='viridis', label='Predicted')
        ax.scatter(X.detach().numpy(), Y.detach().numpy(), y_Exact, c='r', label='Exact Solution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title('Epochs: ' + str(num_epochs) + ', Residual Error: ' + str(residual_error) + ", Samples: " + str(
            num_samples) + "," + str(str_list[j]))

plt.savefig('SolvePDEs_Lagaris_problem_6.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('SolvePDEs_Lagaris_problem_6.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
