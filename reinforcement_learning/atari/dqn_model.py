import torch.nn as nn

class DeepQNetwork(nn.Module):

    def __init__(self, inputs, outputs, hidden_layers, hidden_layer_size):
        super().__init__()

        self.inputs = inputs
        self.outputs = outputs

        self.hidden_layers = hidden_layers
        self.hidden_layer_size = hidden_layer_size

        self.linear_relu_stack = self._create_stack()

    def _create_stack(self):
        if self.hidden_layers <= 1:            
            return nn.Linear(self.inputs, self.outputs)

        layers = list()

        for i in range(self.hidden_layers - 1):
            input_size = self.hidden_layer_size if i != 0 else self.inputs

            layers.append(nn.Linear(input_size, self.hidden_layer_size))            
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_layer_size, self.outputs))
        
        return nn.Sequential(*layers)
    
    def clamp(self):
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)

    def forward(self, x):        
        return self.linear_relu_stack(x)