import torch

class Model(torch.nn.Module):
    
    LAYER_FUNCTIONS = {
        "ReLU": torch.nn.ReLU, "LeakyReLU": torch.nn.LeakyReLU,
        "Tanh": torch.nn.Tanh, "Sigmoid": torch.nn.Sigmoid
    }

    def __init__(self, input_size, output_size, nb_layers=1, max_nb_nodes=10,
                activation_type="LeakyReLU", dropout_rate=0, node_decay=1.5):
        super().__init__()

        self.activation_type = Model.LAYER_FUNCTIONS.get(activation_type, torch.nn.ReLU)
       
        self.input_size = input_size
        self.output_size = output_size
        self.nb_layers = nb_layers

        self.max_nb_nodes = max_nb_nodes
        self.node_decay = node_decay
        self.dropout_rate = dropout_rate

        self.network_stack = self._create_stack()
        self.network_stack.apply(self._init_weights)

    def _get_layer_sizes(self, i):
        input_size = int(self.max_nb_nodes / (self.node_decay ** (i - 1))) \
            if i != 0 else self.input_size
        
        output_size = int(self.max_nb_nodes / (self.node_decay ** i)) \
            if i < self.nb_layers - 1 else self.output_size

        input_size += 1 if input_size == 0 else 0
        output_size += 1 if output_size == 0 else 0      

        return input_size, output_size  

    def _create_stack(self):
        if self.nb_layers == 0:            
            return torch.nn.Sequential(
                torch.nn.Linear(self.input_size, self.output_size),
                torch.nn.Dropout(p=self.dropout_rate),
                self.activation_type()
            )

        layers = list()

        for i in range(self.nb_layers):
            input_size, output_size = self._get_layer_sizes(i)

            layers.append(torch.nn.Linear(input_size, output_size))
            layers.append(torch.nn.Dropout(p=self.dropout_rate))
            
            layers.append(self.activation_type())
        
        return torch.nn.Sequential(*layers)

    def _init_weights(self, mod):
        if isinstance(mod, torch.nn.Linear):
            torch.nn.init.normal_(mod.weight)

    def forward(self, x):
        return self.network_stack(x)
