import json
from trainer.train_runner import run_training

def run_single(
    config: dict,
    train_v,
    val_v,
    test_v,
    train_df,
    val_df,
    test_df,
    training_fn=run_training
):
    """
    Führt einen einzelnen Trainingslauf durch.

    Args:
        config (dict): Konfigurationsparameter
        *_v: Zeitreihenwerte (np.ndarray)
        *_df: Zeitstempel-DataFrames
        training_fn (callable): Trainingsfunktion, default: run_training
    """
    print("[INFO] Running SINGLE training run:")
    print(json.dumps(config, indent=2))

    trainer, test_loader = training_fn(
        config=config,
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

    return trainer, test_loader
