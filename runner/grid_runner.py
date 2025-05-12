import itertools
import json
from trainer.train_runner import run_training

def run_grid(
    grid_config: dict,
    train_v,
    val_v,
    test_v,
    train_df,
    val_df,
    test_df,
    training_fn=run_training  # austauschbar
):
    """
    Führt Grid-Search über alle Parameterkombinationen aus.
    
    Args:
        grid_config (dict): Dictionary mit Parametern und Werte-Listen.
        *_v (np.ndarray): Zeitreihenwerte.
        *_df (pd.DataFrame): Zeitstempel-DataFrames.
        training_fn (callable): Trainingsfunktion.
    """
    keys, values = zip(*grid_config.items())
    total_runs = len(list(itertools.product(*values)))
    print(f"[INFO] Grid Search über {total_runs} Kombinationen")

    for i, v in enumerate(itertools.product(*values)):
        config = dict(zip(keys, v))
        config["patience"] = config.get("patience", 15)  # fallback
        print(f"\n[Run {i+1}/{total_runs}]")
        print(json.dumps(config, indent=2))

        trainer, test_loader = training_fn(
            config=config,
            train_v=train_v,
            val_v=val_v,
            test_v=test_v,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        test_loss, test_metrics = trainer.evaluate(test_loader)
        print(f"[Test Loss]: {test_loss:.6f}")
        for k, v in test_metrics.items():
            print(f"[{k.upper()}]: {v:.6f}")
