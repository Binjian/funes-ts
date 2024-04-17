import numpy as np
import torch

__all__ = ["FeaturePredictionDataset", "OneStepPredictionDataset"]


class FeaturePredictionDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    - idx (int): the index of the feature to be predicted
    """

    def __init__(self, data, time, idx):
        no, seq_len, dim = data.shape
        if dim == 1:
            self.X = torch.FloatTensor(
                data[:, :, 0]
            )
            self.X = np.expand_dims(self.X,axis=-1)
        else:
            self.X = torch.FloatTensor(
                np.concatenate((data[:, :, :idx], data[:, :, (idx + 1) :]), axis=2)
            )
        self.T = torch.LongTensor(time)
        self.Y = torch.FloatTensor(np.reshape(data[:, :, idx], [no, seq_len, 1]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]


class OneStepPredictionDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """

    def __init__(self, data, time):
        no, seq_len, dim = data.shape
        self.X = torch.FloatTensor(data[:, :-1, :])
        # self.T = torch.LongTensor([t - 1 if t == seq_len else t for t in time])
        self.T = torch.DoubleTensor([t for t in time - 1]).long()
        self.Y = torch.FloatTensor(data[:, 1:, :])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]


if __name__ == "__main__":
    pass
