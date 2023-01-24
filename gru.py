from torch import nn


class GRUWeatherProphet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=1) -> None:
        super(GRUWeatherProphet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
