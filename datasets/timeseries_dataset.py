__all__ = ["TimeSeriesDataset"]

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        data,
        timestamps,
        lookback,
        horizon,
        *args,
        **kwargs
    ):
        """
        Dataset zur Vorverarbeitung von Zeitreihen mit Lookback und Horizont.

        Args:
            data (np.ndarray or Tensor): Zeitreihe (z.B. Leistungswerte)
            timestamps (pd.Series): Zeitstempel der Daten
            lookback (int): Anzahl vergangener Zeitschritte (Input-Länge)
            horizon (int): Anzahl vorherzusagender Zeitschritte
        """
        super().__init__(*args, **kwargs)

        self.lookback = lookback
        self.horizon = horizon
        self.time_features = self.create_time_features(timestamps)

        self.data_points = [
            (
                torch.tensor(
                    np.concatenate(
                        [data[i:i+lookback].reshape(-1, 1), self.time_features[i:i+lookback]], axis=1
                    ),
                    dtype=torch.float32
                ),
                torch.tensor(data[i+lookback:i+lookback+horizon], dtype=torch.float32)
            )
            for i in range(len(data) - lookback - horizon + 1)
        ]

    def create_time_features(self, timestamps):
        timestamps = pd.to_datetime(timestamps)
        seconds_in_day = 24 * 60 * 60
        day_seconds = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
        day_sin = np.sin(2 * np.pi * day_seconds / seconds_in_day)
        day_cos = np.cos(2 * np.pi * day_seconds / seconds_in_day)

        day_of_year = timestamps.dt.dayofyear
        year_sin = np.sin(2 * np.pi * day_of_year / 365)
        year_cos = np.cos(2 * np.pi * day_of_year / 365)

        return np.stack([day_sin, day_cos, year_sin, year_cos], axis=1).astype(np.float32)

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        return self.data_points[idx]
