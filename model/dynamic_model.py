import torch
import torch.nn as nn
from base import BaseModel  # Adjust this import if needed based on your structure

# Activation function mapping
def get_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {name}")

class DynamicModel(BaseModel):
    def __init__(self, input_dim=1, output_dim=1, hidden_layers=2, hidden_units=32,
                 hidden_activation='relu', output_activation='linear'):
        super().__init__()

        assert 1 <= hidden_layers <= 5, "Hidden layers must be between 1 and 5"
        assert 1 <= hidden_units <= 100, "Hidden units per layer must be between 1 and 100"

        layers = []
        in_dim = input_dim

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(get_activation(hidden_activation))
            in_dim = hidden_units

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(get_activation(output_activation))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
