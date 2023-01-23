import torch
from torch import nn


class ClassicWeatherProphet(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(ClassicWeatherProphet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
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
