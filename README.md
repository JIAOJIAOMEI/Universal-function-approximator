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