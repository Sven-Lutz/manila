import yaml
import argparse
import itertools
from test_run import run_training, run_sweep
from wandb_login import login_wandb
import pickle
import numpy as np
import pandas as pd
import wandb

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

def run_single(config, train_df, val_df, test_df):
    lookback = config['lookback']
    horizon = config['horizon']
    batch_size = config['batch_size']
    epochs = config['epochs']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    patience = config['patience']

    train_v = train_df["v"].to_numpy(dtype=np.float32)
    val_v = val_df["v"].to_numpy(dtype=np.float32)
    test_v = test_df["v"].to_numpy(dtype=np.float32)

    wandb.init(project="manila-forecasting", config=config)

    trainer, test_loader = run_training(
        lookback, horizon, batch_size, epochs, patience,
        train_v, val_v, test_v,
        train_df, val_df, test_df,
        debug=False
    )

    loss, metrics = trainer.evaluate(test_loader)
    wandb.log({"test_loss": loss, **metrics})
    wandb.finish()

def run_grid(grid_config, train_df, val_df, test_df):
    keys, values = zip(*grid_config.items())
    for i, v in enumerate(itertools.product(*values)):
        params = dict(zip(keys, v))
        print(f"Running experiment {i}: {params}")
        run_single({**params, "patience": 15}, train_df, val_df, test_df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    login_wandb()

    data_path = "Data"
    filename = "D1087183267_ts_dict.pkl"
    train_df, val_df, test_df = load_data(data_path, filename)

    if config["mode"] == "single":
        run_single(config["single_run"], train_df, val_df, test_df)

    elif config["mode"] == "grid":
        run_grid(config["grid_options"], train_df, val_df, test_df)

    elif config["mode"] == "sweep":
        run_sweep(config_path=args.config, count=30)

    else:
        raise ValueError("Unknown mode in config")

if __name__ == "__main__":
    main()
