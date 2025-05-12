# main.py

# Fix: Ermöglicht Imports aus Projektstruktur auch bei direktem Start
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import argparse
import yaml
import json

# Eigene Module
from runner.grid_runner import run_grid
from runner.single_runner import run_single
from utils.data_loader import load_data
from utils.wandb_utils import login_wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Run forecasting pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to folder containing data")
    parser.add_argument("--filename", type=str, required=True, help="Pickle file containing time series dict")
    return parser.parse_args()


def main():
    args = parse_args()

    # Lade Konfiguration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    login_wandb()

    # Lade Daten
    train_df, val_df, test_df = load_data(args.data_path, args.filename)
    train_v = train_df["v"].to_numpy(dtype=float)
    val_v = val_df["v"].to_numpy(dtype=float)
    test_v = test_df["v"].to_numpy(dtype=float)

    mode = config.get("mode", "grid")

    if mode == "single":
        print("[INFO] Run mode: SINGLE")
        print(json.dumps(config["single_run"], indent=2))
        run_single(
            config["single_run"],
            train_v, val_v, test_v,
            train_df, val_df, test_df
        )

    elif mode == "grid":
        print("[INFO] Run mode: GRID SEARCH")
        print(json.dumps(config["grid_options"], indent=2))
        run_grid(
            config["grid_options"],
            train_v, val_v, test_v,
            train_df, val_df, test_df
        )

    else:
        raise ValueError(f"[ERROR] Unknown mode: {mode}")


if __name__ == "__main__":
    main()
