# @Author  : Mei Jiaojiao
# @Time    : 2024/4/8 13:18
# @Software: PyCharm
# @File    : SolveODEs_Lagaris_problem_4.py


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


class ODESolver(torch.nn.Module):

    def __init__(self, numHiddenNodes):
        super(ODESolver, self).__init__()
        self.fully_connected_1 = torch.nn.Linear(in_features=1, out_features=numHiddenNodes)
        self.fully_connected_2 = torch.nn.Linear(in_features=numHiddenNodes, out_features=2)

    def forward(self, x, activation_Function):
        first_layer_output = self.fully_connected_1(x)
        first_layer_output = activation_Function(first_layer_output)
        y = self.fully_connected_2(first_layer_output)
        return y


def f1_final_output(x, n1_out):
    return x * n1_out


def f2_final_output(x, n2_out):
    return 1 + (x * n2_out)


def f1_derivative(x, n1_out, d1ndx):
    return n1_out + (x * d1ndx)


def f2_derivative(x, n2_out, d2ndx):
    return n2_out + (x * d2ndx)


def Lagaris_problem_4_eq1(x, f1_trial, f2_trial, df1_trial):
    LHS = df1_trial
    RHS = torch.cos(x) + (f1_trial ** 2 + f2_trial) - (1 + x ** 2 + torch.sin(x) ** 2)
    return LHS - RHS


def Lagaris_problem_4_eq2(x, f1_trial, f2_trial, df2_trial):
    LHS = df2_trial
    RHS = 2 * x - ((1 + x ** 2) * torch.sin(x)) + (f1_trial * f2_trial)
    return LHS - RHS


def solution_to_Lagaris_problem_4_eq1(x):
    return torch.sin(x)


def solution_to_Lagaris_problem_4_eq2(x):
    return 1 + x ** 2


def train(neural_network, data_loader, loss_function, optimiser, num_Epochs, activationFn):
    cost_list = []
    neural_network.train(True)  # set module in training mode
    for epoch in range(num_Epochs):
        for batch in data_loader:
            n_out = neural_network.forward(batch, activationFn)
            n1_out, n2_out = torch.split(n_out, split_size_or_sections=1, dim=1)
            # Get df1 / dx
            dn1_dx = grad(n1_out, batch, torch.ones_like(n1_out), retain_graph=True, create_graph=True)[0]
            # Get df2 / dx
            dn2_dx = grad(n2_out, batch, torch.ones_like(n2_out), retain_graph=True, create_graph=True)[0]

            # Get value of trial solution f1(x)
            f1_trial = f1_final_output(batch, n1_out)
            # Get df1 / dx
            df1_trial = f1_derivative(batch, n1_out, dn1_dx)

            # Get value of trial solution f2(x)
            f2_trial = f2_final_output(batch, n2_out)
            # Get df2 / dx
            df2_trial = f2_derivative(batch, n2_out, dn2_dx)

            # Get LHS of differential equation D(x) = 0
            diff_eq1 = Lagaris_problem_4_eq1(batch, f1_trial, f2_trial, df1_trial)
            diff_eq2 = Lagaris_problem_4_eq2(batch, f1_trial, f2_trial, df2_trial)

            cost = loss_function(diff_eq1, torch.zeros_like(diff_eq1)) + loss_function(diff_eq2, torch.zeros_like(diff_eq2))

            cost.backward()  # perform backpropagation
            optimiser.step()  # perform parameter optimisation
            optimiser.zero_grad()  # reset gradients to zero

        cost_list.append(cost.detach().numpy())  # store cost of each epoch
    neural_network.train(False)  # set module out of training mode
    return cost_list


x_range = [0, 3]
num_samples = 100
batch_size = 20
learning_rate = 1e-3
num_hidden_nodes = 20

network = ODESolver(numHiddenNodes=num_hidden_nodes)

train_set = DataSet(xRange=x_range, numSamples=num_samples)
loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
lossFn = torch.nn.MSELoss()

optimiser_network = torch.optim.Adam(network.parameters(), lr=learning_rate)

activationFn_list = [torch.tanh, torch.relu, torch.sigmoid, torch.nn.functional.leaky_relu, torch.nn.functional.elu,
                     torch.nn.functional.hardswish]
num_epochs_list = [100, 500, 1000]

rows = len(activationFn_list)
cols = len(num_epochs_list)

num_samples_test = 2 * num_samples
fig, axs = plt.subplots(rows, cols, figsize=(50, 50))
for i, activationFn in enumerate(activationFn_list):
    for j, num_epochs in enumerate(num_epochs_list):
        cost_List = train(network, loader, lossFn, optimiser_network, num_epochs, activationFn)
        x = torch.linspace(x_range[0], x_range[1], num_samples_test).view(-1, 1)
        y1_Exact = solution_to_Lagaris_problem_4_eq1(x).detach().numpy()
        y2_Exact = solution_to_Lagaris_problem_4_eq2(x).detach().numpy()
        y1_Out, y2_Out = torch.split(network.forward(x, activationFn), split_size_or_sections=1, dim=1)
        y1_Out = y1_Out.detach().numpy()
        y2_Out = y2_Out.detach().numpy()
        x = x.detach().numpy()

        axs[i, j].plot(x, y1_Exact, 'r', label='Exact f1(x)')
        axs[i, j].plot(x, y2_Exact, 'b', label='Exact f2(x)')
        axs[i, j].plot(x, f1_final_output(x, y1_Out), 'g', label='Approx f1(x)')
        axs[i, j].plot(x, f2_final_output(x, y2_Out), 'k', label='Approx f2(x)')

        axs[i, j].set_title('Activation Function: ' + activationFn.__name__ + ', Epochs: ' + str(num_epochs))
        axs[i, j].legend()
plt.savefig('SolveODEs_Lagaris_problem_4.pdf',dpi=300,bbox_inches='tight',pad_inches=0.1)
plt.savefig('SolveODEs_Lagaris_problem_4.png',dpi=300,bbox_inches='tight',pad_inches=0.1)
plt.show()
