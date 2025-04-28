import os
import pickle
import pandas as pd
import numpy as np
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import yaml
import itertools

from wandb_login import login_wandb
from metrics import mean_absolute_error, root_mean_squared_error
from models import BasicLSTM
from datasets import TimeSeriesDataset
from trainer import BasicTrainer

login_wandb()

def load_data(data_path, filename):
    pv_ts_dict = pickle.load(open(f"{data_path}/{filename}", 'rb'))
    df = pv_ts_dict['data'].copy()
    df['ts'] = pd.to_datetime(df['ts'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin').dt.tz_localize(None)

    latest_date = df['ts'].max()
    cutoff = latest_date - pd.DateOffset(months=6)
    val_cutoff = cutoff - pd.DateOffset(months=6)

    return (
        df[df['ts'] < val_cutoff],
        df[(df['ts'] >= val_cutoff) & (df['ts'] < cutoff)],
        df[df['ts'] >= cutoff]
    )



def run_training(
        lookback, horizon, batch_size, epochs, patience,
        train_v, val_v, test_v,
        train_df, val_df, test_df,
        debug=True
    ):
    
    wandb.init(
        project="manila-forecasting",
        config={
            "batch_size": batch_size,
            "epochs": epochs,
            "lookback": lookback,
            "horizon": horizon
        }
    )

    assert len(train_v) > 0
    assert len(val_v) > 0
    assert len(test_v) > 0

    if debug:
        lookback = 4
        horizon = 2
        batch_size = 32
        train_v = train_v[:10]
        val_v = val_v[:10]
        test_v = test_v[:10]

    train_dataset = TimeSeriesDataset(train_v, train_df['ts'], lookback, horizon)
    val_dataset = TimeSeriesDataset(val_v, val_df['ts'], lookback, horizon)
    test_dataset = TimeSeriesDataset(test_v, test_df['ts'], lookback, horizon)

    train_dataloader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = t.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = t.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BasicLSTM(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        output_size=horizon,
        dropout=0.1,
    )
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    wandb.config.update({"learning_rate": optimizer.param_groups[0]["lr"]})

    metrics = {
        'mae': mean_absolute_error,
        'rmse': root_mean_squared_error
    }

    my_trainer = BasicTrainer(
        model=model,
        device='cuda' if t.cuda.is_available() else 'cpu',
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        log_dir='./logs',
        checkpoint_dir='./checkpoints'
    )

    print(f'Start run with: batch size {batch_size}, epochs {epochs} and patience {patience}.')

    training_results = my_trainer.run(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        num_epochs=epochs,
        patience=patience,
        checkpoint=None
    )

    return my_trainer, test_dataloader



def log_predictions_wandb(model, test_loader, device):
    """
    Logs the predictions vs. ground truth directly in Weights & Biases (W&B) as a table.
    """
    model.eval()
    preds, true_values = [], []

    with t.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy()
            
            true_values.extend(targets.cpu().numpy().flatten())  
            preds.extend(outputs.flatten())  

    print(f"Ground Truth (erste 10 Werte): {true_values[:10]}")
    print(f"Predictions (erste 10 Werte): {preds[:10]}")

    data = [[i, float(truth), float(pred)] for i, (truth, pred) in enumerate(zip(true_values, preds))]

    print(f"Data für W&B (erste 10 Zeilen): {data[:10]}")  # Debugging-Print

    table = wandb.Table(data=data, columns=["Index", "Ground Truth", "Prediction"])

    wandb.log({
        "Predictions vs Ground Truth": wandb.plot.line(
            table, "Index", "Prediction", title="Predictions"
        ),
        "Ground Truth Line": wandb.plot.line(
            table, "Index", "Ground Truth", title="Ground Truth"
        )
    })


def sweep_train(config=None):
    with wandb.init(config=config):
        cfg = wandb.config

        data_path = "/home/jupyter-leo.jessat/manila/data/processed-data"
        filename = "D1087183267_ts_dict.pkl"
        train_df, val_df, test_df = load_data(data_path, filename)

        train_v = train_df["v"].to_numpy(dtype=float)
        val_v = val_df["v"].to_numpy(dtype=float)
        test_v = test_df["v"].to_numpy(dtype=float)

        trainer, test_loader = run_training(
            lookback=cfg.lookback,
            horizon=cfg.horizon,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            patience=cfg.patience,
            train_v=train_v,
            val_v=val_v,
            test_v=test_v,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            debug=False
        )

        loss, metrics = trainer.evaluate(test_loader)
        wandb.log({"test_loss": loss, **metrics})


def run_sweep(config_path="config.yaml", count=20):
    login_wandb()
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
        sweep_config = full_config["sweep_config"]

    sweep_id = wandb.sweep(sweep_config, project="manila-forecasting")
    wandb.agent(sweep_id, function=sweep_train, count=count)

def main():
    """
    Loads data, creates train/test splits, runs training and logs the predictions.
    """
    data_path = "Data"
    filename = "D1087183267_ts_dict.pkl"




    lookback = 4 * 24
    horizon = 4 * 6
    batch_size = 32
    epochs = 250
    patience = 15

    pv_ts_dict = pickle.load(open(os.path.join(data_path, filename), 'rb'))
    df = pv_ts_dict['data'].copy()
    df['ts'] = pd.to_datetime(df['ts'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin').dt.tz_localize(None)

    latest_date = df['ts'].max()
    cutoff = latest_date - pd.DateOffset(months=6)
    val_cutoff = cutoff - pd.DateOffset(months=6) 

    train_df = df[df['ts'] < val_cutoff]
    val_df = df[(df['ts'] >= val_cutoff) & (df['ts'] < cutoff)]
    test_df = df[df['ts'] >= cutoff]

    train_v = train_df["v"].to_numpy(dtype=np.float32)
    val_v = val_df["v"].to_numpy(dtype=np.float32)
    test_v = test_df["v"].to_numpy(dtype=np.float32)
    my_trainer, test_dataloader = run_training(
        lookback, horizon, batch_size, epochs, patience,
        train_v, val_v, test_v,
        train_df, val_df, test_df,
        debug=False
    )


    test_loss, test_metrics = my_trainer.evaluate(test_dataloader)

    print(f'Test loss: {test_loss: .15f}')
    for k, v in test_metrics.items():
        print(f'Test metric {k}: {v: .15f}', end=', ')
    print()

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    log_predictions_wandb(my_trainer.model.to(device), test_dataloader, device)

    wandb.finish()
    print('Done.')
    print('Exit training.')

if __name__ == '__main__':
    main()
