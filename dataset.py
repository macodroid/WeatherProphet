from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, feature, labels, window_size):
        self.feature = feature
        self.labels = labels

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.labels[idx].reshape(-1, 1)
