import torch
import torch.nn as nn
from torch import Tensor

class BasicLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.0,
    ):
        """
        Einfaches LSTM-Modell zur Zeitreihen-Vorhersage.

        Args:
            input_size (int): Anzahl Input-Features (z. B. 1 Wert + 4 Zeitfeatures = 5)
            hidden_size (int): Größe des LSTM-Hidden-States
            num_layers (int): Anzahl der LSTM-Schichten
            output_size (int): Anzahl vorhergesagter Werte (z. B. Horizon)
            dropout (float): Dropout zwischen LSTM-Schichten (nur bei num_layers > 1)
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        lstm_out, _ = self.lstm(x, (h_0, c_0))
        last_output = lstm_out[:, -1, :]  # Nur letzter Zeitschritt
        return self.fc(last_output)
