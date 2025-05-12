import os
import copy
import torch
from torch.utils.data import DataLoader
from typing import Callable, Dict, Optional, Tuple, Any

class BasicTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or {}

        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _train_step(self, inputs, targets):
        self.model.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        metrics_result = {k: metric(outputs, targets).item() for k, metric in self.metrics.items()}
        return loss.item(), metrics_result

    def _eval_step(self, inputs, targets):
        self.model.eval()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            metrics_result = {k: metric(outputs, targets).item() for k, metric in self.metrics.items()}
        return loss.item(), metrics_result

    def train_epoch(self, loader: DataLoader):
        total_loss = 0.0
        total_metrics = {k: 0.0 for k in self.metrics}
        total_samples = len(loader)

        for inputs, targets in loader:
            loss, metrics_result = self._train_step(inputs, targets)
            total_loss += loss * inputs.size(0)
            for k in self.metrics:
                total_metrics[k] += metrics_result[k] * inputs.size(0)

        avg_loss = total_loss / total_samples
        avg_metrics = {k: v / total_samples for k, v in total_metrics.items()}
        return avg_loss, avg_metrics

    def evaluate(self, loader: DataLoader):
        total_loss = 0.0
        total_metrics = {k: 0.0 for k in self.metrics}
        total_samples = len(loader)

        self.model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                loss, metrics_result = self._eval_step(inputs, targets)
                total_loss += loss * inputs.size(0)
                for k in self.metrics:
                    total_metrics[k] += metrics_result[k] * inputs.size(0)

        avg_loss = total_loss / total_samples
        avg_metrics = {k: v / total_samples for k, v in total_metrics.items()}
        return avg_loss, avg_metrics

    def run(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        patience: Optional[int] = None
    ) -> Dict[str, Any]:

        best_val_loss = float("inf")
        best_model_state = copy.deepcopy(self.model.state_dict())
        no_improvement = 0
        history = {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []}

        for epoch in range(num_epochs):
            print(f"[Epoch {epoch + 1}/{num_epochs}]")

            train_loss, train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_metrics"].append(train_metrics)

            print(f"Train Loss: {train_loss:.6f}", end="  ")
            for k, v in train_metrics.items():
                print(f"{k.upper()}: {v:.6f}", end="  ")
            print()

            if val_loader:
                val_loss, val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_metrics"].append(val_metrics)

                print(f"Val Loss: {val_loss:.6f}", end="  ")
                for k, v in val_metrics.items():
                    print(f"{k.upper()}: {v:.6f}", end="  ")
                print()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if patience and no_improvement >= patience:
                        print("Early stopping triggered.")
                        break

        self.model.load_state_dict(best_model_state)
        return {"model": self.model, "history": history}