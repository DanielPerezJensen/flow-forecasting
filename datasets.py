import torch
import numpy as np
from torch.utils.data import Dataset


class RiverFlowDataset(Dataset):
    def __init__(self, df, scaler=None):
        x = df.iloc[:, 2:].values
        y = df.iloc[:, 1].values[..., np.newaxis]

        if scaler:
            x = scaler.fit_transform(x)
            y = scaler.fit_transform(y)

            self.scaler = scaler

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]