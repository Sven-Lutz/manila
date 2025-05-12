import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.basic_lstm import BasicLSTM
from datasets.timeseries_dataset import TimeSeriesDataset
from trainer.trainer import BasicTrainer
from metrics import mean_absolute_error, root_mean_squared_error
import json
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

def run_training(
    config: dict,
    train_v,
    val_v,
    test_v,
    train_df,
    val_df,
    test_df,
):
    """
    Führt das vollständige Training mit Konfiguration und Daten durch.
    """

    print("[TRAINING CONFIG]")
    print(json.dumps(config, indent=2))

    lookback = config["lookback"]
    horizon = config["horizon"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    patience = config["patience"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]

    # Dataset
    train_dataset = TimeSeriesDataset(train_v, train_df['ts'], lookback, horizon)
    val_dataset = TimeSeriesDataset(val_v, val_df['ts'], lookback, horizon)
    test_dataset = TimeSeriesDataset(test_v, test_df['ts'], lookback, horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BasicLSTM(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=horizon,
        dropout=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    criterion = nn.MSELoss()

    metrics = {
        "mae": mean_absolute_error,
        "rmse": root_mean_squared_error
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = BasicTrainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler if isinstance(scheduler, _LRScheduler) else None,
        metrics=metrics,
        log_dir="./logs",
        checkpoint_dir="./checkpoints"
    )


    print(f"[INFO] Start training: {epochs} epochs, batch size {batch_size}, patience {patience}")
    trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        patience=patience
    )

    return trainer, test_loader
