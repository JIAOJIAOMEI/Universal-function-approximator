# Universal function approximator
Neural networks are universal function approximators. 
Given enough neurons and layers, a neural network can approximate **any** function.
(I don't really believe this, but I agree that neural networks are very powerful function approximators.)
If you are interested in the proof, you can read the paper [Multilayer feedforward networks are universal approximators](https://github.com/JIAOJIAOMEI/Universal-function-approximator-and-PINNs/blob/main/1989-Multilayer%20feedforward%20networks%20are%20universal%20approximators.pdf).

![FunctionApproximation.png](FunctionApproximation.png)

```python

class Fitter(torch.nn.Module):

    def __init__(self, numHiddenNodes):
        super(Fitter, self).__init__()
        self.fully_connected_1 = torch.nn.Linear(in_features=1, out_features=numHiddenNodes)
        self.fully_connected_2 = torch.nn.Linear(in_features=numHiddenNodes, out_features=1)

    def forward(self, x, activation_Function):
        h = activation_Function(self.fully_connected_1(x))
        # Linear activation function used on the outer layer
        y = self.fully_connected_2(h)
        return y

```

# Fourier feature mapping

Fourier's transformation is indeed function approximation.
It can approximate any function with a series of sine and cosine functions.

**Since that Fourier's transformation and neural networks are both function approximators, can we combine them together?**

if you want more details, you can read the paper [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://github.com/JIAOJIAOMEI/Universal-function-approximator-and-PINNs/blob/main/Fourier%20Features%20Let%20Networks%20Learn%20High%20Frequency%20Functions%20in%20Low%20Dimensional%20Domains.pdf).

![Fourier_FunctionApproximation.png](Fourier_FunctionApproximation.png)

```python

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

```

# Different optimizers

![Optimiser_Compare.png](Optimiser_Compare.png)

# Solving ODEs with PINNs Example

The idea is based on this paper [Artificial Neural Networks for Solving ODEs and PDES](https://github.com/JIAOJIAOMEI/Universal-function-approximator-and-PINNs/blob/main/1998-Artificial%20Neural%20Networks%20for%20Solving%20ODEs%20and%20PDES.pdf),
the code implementation is based on this [Neural Networks for Solving Differential Equations](https://github.com/JIAOJIAOMEI/Universal-function-approximator-and-PINNs/blob/main/main%20reference%20for%20this%20project.pdf), I made some modifications.


$$
\begin{equation}
\frac{d f(x)}{d x}=x^3+2 x+x^2 \frac{1+3 x^2}{1+x+x^3}-\left(x+\frac{1+3 x^2}{1+x+x^3}\right) f(x)
\end{equation}
$$


for $x \in[0,2]$, with $f(0)=1$.

The true solution is 


$$
\begin{equation}
f(x)=\frac{e^{-\frac{x^2}{2}}}{1+x+x^3}+x^2
\end{equation}
$$

I compared normal networks and fourier networks, the results are shown below.

![FirstOrderODE_Lagaris_problem_1.png](FirstOrderODE_Lagaris_problem_1.png)

```python
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import grad


class DataSet(torch.utils.data.Dataset):

    def __init__(self, xRange, numSamples):
        self.InputData = torch.linspace(xRange[0], xRange[1], numSamples, requires_grad=True).view(-1, 1)

    def __len__(self):
        return len(self.InputData)

    def __getitem__(self, idx):
        return self.InputData[idx]


def fourier_feature_mappping_function(x):
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class Fourier_Fitter(torch.nn.Module):

    def __init__(self, numHiddenNodes):
        super(Fourier_Fitter, self).__init__()
        self.fully_connected_1 = torch.nn.Linear(in_features=2, out_features=numHiddenNodes)
        self.fully_connected_2 = torch.nn.Linear(in_features=numHiddenNodes, out_features=1)

    def forward(self, x, activation_Function):
        first_layer_output = self.fully_connected_1(fourier_feature_mappping_function(x))
        first_layer_output = activation_Function(first_layer_output)
        y = self.fully_connected_2(first_layer_output)
        return y


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
    """
    x range: [0,2]
    f(0) = 1
    Trial solution to Lagaris problem 1: f(x) = 1 + xN(x)
    """
    return 1 + x * n_out


def function_derivative(x, n_out, dndx):
    """
    Derivative of a trial solution to Lagaris problem 1: f'(x) = N(x) + xN'(x)
    """
    return n_out + x * dndx


def Lagaris_problem_1(x, f_trial, df_trial):
    RHS = x ** 3 + 2 * x + (x ** 2 * ((1 + 3 * x ** 2) / (1 + x + x ** 3)))
    LHS = df_trial + ((x + (1 + 3 * (x ** 2)) / (1 + x + x ** 3)) * f_trial)
    return LHS - RHS


def solution_to_Lagaris_problem_1(x):
    y = (torch.exp(-(x ** 2) / 2) / (1 + x + x ** 3)) + x ** 2
    return y


def train(neural_network, data_loader, loss_function, optimiser, num_Epochs, activationFn):
    cost_list = []
    neural_network.train(True)  # set module in training mode
    for epoch in range(num_Epochs):
        for batch in data_loader:
            n_out = neural_network.forward(batch, activationFn)
            dn_dx = grad(n_out, batch, torch.ones_like(n_out), retain_graph=True)[0]

            # Get value of trial solution f(x)
            f_trial = final_output(batch, n_out)
            # Get df / dx
            df_trial = function_derivative(batch, n_out, dn_dx)
            # Get LHS of differential equation D(x) = 0
            diff_eq = Lagaris_problem_1(batch, f_trial, df_trial)

            cost = loss_function(diff_eq, torch.zeros_like(diff_eq))
            # torch.zeros_like(x) creates a tensor the same shape as x, filled with 0's
            cost.backward()  # perform backpropagation
            optimiser.step()  # perform parameter optimisation
            optimiser.zero_grad()  # reset gradients to zero

        cost_list.append(cost.detach().numpy())  # store cost of each epoch
    neural_network.train(False)  # set module out of training mode
    return cost_list


x_range = [0, 2]
num_samples = 100
batch_size = 20
learning_rate = 1e-3
num_epochs = 50
num_hidden_nodes = 10

network = Fitter(numHiddenNodes=num_hidden_nodes)
Fourier_network = Fourier_Fitter(numHiddenNodes=num_hidden_nodes)

train_set = DataSet(xRange=x_range, numSamples=num_samples)
loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
lossFn = torch.nn.MSELoss()

optimiser_network = torch.optim.SGD(network.parameters(), lr=learning_rate)
optimiser_Fourier_network = torch.optim.SGD(Fourier_network.parameters(), lr=learning_rate)

activationFn_list = [torch.tanh, torch.relu, torch.sigmoid, torch.nn.functional.leaky_relu, torch.nn.functional.elu,
                     torch.nn.functional.hardswish]

rows = len(activationFn_list)
cols = 2

fig, axs = plt.subplots(rows, cols, figsize=(50, 50))
for i, activationFn in enumerate(activationFn_list):
    cost_List = train(network, loader, lossFn, optimiser_network, num_epochs, activationFn)
    x = torch.linspace(x_range[0], x_range[1], num_samples).view(-1, 1)
    y_Exact = solution_to_Lagaris_problem_1(x).detach().numpy()
    y_Out = network.forward(x, activationFn).detach().numpy()
    x = x.detach().numpy()

    axs[i, 0].plot(x, y_Exact, 'b.', label='Exact')
    axs[i, 0].plot(x, final_output(x, y_Out), 'r.', label='Approx')
    axs[i, 0].set_title(f'Activation Function: {activationFn.__name__}, Neural Fitter,{num_epochs} Epochs')
    axs[i, 0].legend()

    cost_List = train(Fourier_network, loader, lossFn, optimiser_Fourier_network, num_epochs, activationFn)
    x = torch.linspace(x_range[0], x_range[1], num_samples).view(-1, 1)
    y_Exact = solution_to_Lagaris_problem_1(x).detach().numpy()
    y_Out = Fourier_network.forward(x, activationFn).detach().numpy()
    x = x.detach().numpy()

    axs[i, 1].plot(x, y_Exact, 'b.', label='Exact')
    axs[i, 1].plot(x, final_output(x, y_Out), 'r.', label='Approx')
    axs[i, 1].set_title(f'Activation Function: {activationFn.__name__}, Fourier Fitter,{num_epochs} Epochs')
    axs[i, 1].legend()

plt.savefig('FirstOrderODE_Lagaris_problem_1.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('FirstOrderODE_Lagaris_problem_1.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
    
    ```
