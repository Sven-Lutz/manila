import argparse
import yaml
import numpy as np
import torch
from utils.data_loader import load_data
from utils.wandb_utils import login_wandb, log_predictions_wandb
from trainer.train_runner import run_training

def parse_args():
    parser = argparse.ArgumentParser(description="Testlauf für Training und Logging")
    parser.add_argument("--config", type=str, default="config.yaml", help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--data_path", type=str, required=True, help="Pfad zum Datenordner")
    parser.add_argument("--filename", type=str, required=True, help="Pickle-Dateiname")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    run_cfg = config["single_run"]

    login_wandb()

    train_df, val_df, test_df = load_data(args.data_path, args.filename)
    train_v = train_df["v"].to_numpy(dtype=np.float32)
    val_v = val_df["v"].to_numpy(dtype=np.float32)
    test_v = test_df["v"].to_numpy(dtype=np.float32)

    trainer, test_loader = run_training(
        config=run_cfg,
        train_v=train_v,
        val_v=val_v,
        test_v=test_v,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    test_loss, test_metrics = trainer.evaluate(test_loader)
    print(f"[Test Loss]: {test_loss:.6f}")
    for k, v in test_metrics.items():
        print(f"[{k.upper()}]: {v:.6f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_predictions_wandb(trainer.model.to(device), test_loader, device)

if __name__ == "__main__":
    main()
