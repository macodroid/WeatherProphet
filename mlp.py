import torch
from torch import nn


class ClassicWeatherProphet(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(ClassicWeatherProphet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Initialize weights
        def init_weights(m):
            if type(m) in [nn.Linear]:
                nn.init.kaiming_uniform_(m.weight)

        # Apply initialization
        self.model.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.model(x)
