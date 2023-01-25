import torch
from torch import nn


class GRUWeatherProphet(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, batch_first=True
    ):
        super(GRUWeatherProphet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        def init_weights(m):
            if type(m) in [nn.Linear]:
                nn.init.kaiming_uniform_(m.weight)

        # Apply initialization
        self.fc.apply(init_weights)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).cuda()
        out, _ = self.gru(x, h0)
        r_out = out[:, -1, :]
        out = self.fc(r_out)
        return out
