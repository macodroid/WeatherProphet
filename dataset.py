from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, temperture_mean, temperture_std):
        self.data = data
        self.window_size = window_size
        self.temperture_mean = temperture_mean
        self.temperture_std = temperture_std

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        window = self.data[idx: idx + self.window_size]
        # label = (
        #     self.data[idx + self.window_size][-1] * self.temperture_std
        # ) + self.temperture_mean
        label = self.data[idx + self.window_size][-1]
        return window, label.reshape(1,)
