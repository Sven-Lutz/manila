import os
import wandb
import torch
from dotenv import load_dotenv
from wandb.errors import UsageError  

def login_wandb():
    """
    Meldet sich bei Weights & Biases mit dem API-Key aus der .env-Datei an.
    """
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")

    if api_key:
        try:
            wandb.login(key=api_key)
            print("[W&B] Login erfolgreich.")
        except UsageError as e:  # ✅ Korrekt verwendet
            print(f"[W&B] Login fehlgeschlagen: {e}")
    else:
        print("[W&B] Kein WANDB_API_KEY gefunden – bitte .env-Datei prüfen.")

def log_predictions_wandb(model, test_loader, device):
    """
    Loggt Vorhersagen und Ground Truth als Linienplot in W&B.

    Args:
        model (torch.nn.Module): Trainiertes Modell.
        test_loader (DataLoader): Test-Daten.
        device (str): "cuda" oder "cpu".
    """
    model.eval()
    model.to(device)

    preds, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch).cpu().numpy()
            preds.extend(output.flatten())
            targets.extend(y_batch.cpu().numpy().flatten())

    table = wandb.Table(
        data=[[i, float(t), float(p)] for i, (t, p) in enumerate(zip(targets, preds))],
        columns=["Index", "Ground Truth", "Prediction"]
    )

    wandb.log({
        "Prediction Plot": wandb.plot.line(table, "Index", "Prediction", title="Predictions"),
        "Ground Truth Plot": wandb.plot.line(table, "Index", "Ground Truth", title="Ground Truth")
    })
