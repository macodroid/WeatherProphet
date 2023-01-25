from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, feature, labels, window_size):
        self.feature = feature
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.feature) - self.window_size

    def __getitem__(self, idx):
        features = self.feature[idx: idx + self.window_size]
        labels = self.labels[idx + self.window_size].reshape(-1, 1)
        return features, labels
