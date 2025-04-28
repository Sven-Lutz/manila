import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from typing import Callable, Dict, Optional, Tuple, Any, List
import wandb

class BasicTrainer:
    """
    A class to train time series models.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]] = None,
        log_dir: Optional[str] = './logs',
        checkpoint_dir: Optional[str] = './checkpoints',
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the TimeSeriesTrainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            device (torch.device): The device to run the training on.
            criterion (Callable): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (Optional[optim.lr_scheduler._LRScheduler]): The learning rate scheduler.
            metrics (Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]): A dictionary of metric functions.
            log_dir (Optional[str]): The directory to save logs.
            checkpoint_dir (Optional[str]): The directory to save model checkpoints.
        """
        self._checkpoint_dir = None
        self._log_dir = None
        
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics or {}

        self._log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir


    
    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Performs a single training step.
        
        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels.

        Returns:
            Tuple[float, Dict[str, float]]: The loss and a dictionary of metric results.
        """
        if not isinstance(inputs, torch.Tensor):
            print(f"Inputs should be a torch.Tensor, but got {type(inputs)}")
        if not isinstance(targets, torch.Tensor):
            print(f"Targets should be a torch.Tensor, but got {type(targets)}")

        self.model.train()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.set_grad_enabled(True):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        step_metrics = {name: metric(outputs, targets).item() for name, metric in self.metrics.items()}

        return loss.item(), step_metrics
    
    def _eval_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Performs a single evaluation step.
        
        Args:
            inputs (torch.Tensor): The input data.
            targets (torch.Tensor): The target labels.

        Returns:
            Tuple[float, Dict[str, float]]: The loss and a dictionary of metric results.
        """
        if not isinstance(inputs, torch.Tensor):
            print(f"Inputs should be a torch.Tensor, but got {type(inputs)}")
        if not isinstance(targets, torch.Tensor):
            print(f"Targets should be a torch.Tensor, but got {type(targets)}")

        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        metrics_result = {name: metric(outputs, targets).item() for name, metric in self.metrics.items()}
        return loss.item(), metrics_result
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Trains the model for one epoch.
        
        Args:
            train_loader (DataLoader): The DataLoader for the training data.

        Returns:
            Tuple[float, Dict[str, float]]: The average loss and a dictionary of average metric results.
        """
        running_loss = 0.0
        metrics_result = {k: 0.0 for k in self.metrics}
        total_samples = len(train_loader.dataset)

        for inputs, targets in train_loader:
            loss, step_metrics = self._train_step(inputs, targets)

            running_loss += loss * inputs.size(0)
            for name, metric_value in step_metrics.items():
                metrics_result[name] += metric_value * inputs.size(0)

            if self.scheduler is not None:
                self.scheduler.step()

        epoch_loss = running_loss / total_samples
        epoch_metrics = {k: v / total_samples for k, v in metrics_result.items()}

        msg = 'Training finished with loss:'
        print(f"{msg:<30} {epoch_loss: .10f} and metrics", end=" ")
        for k, v in epoch_metrics.items():
            print(f"{k}: {v: .10f}", end=", ")
        print()

        return epoch_loss, epoch_metrics
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the model on the validation set.
        
        Args:
            val_loader (DataLoader): The DataLoader for the validation data.
        
        Returns:
            Tuple[float, Dict[str, float]]: The average loss and a dictionary of average metric results.
        """
        running_loss = 0.0
        metrics_result = {k: 0.0 for k in self.metrics}
        total_samples = len(val_loader.dataset)

        self.model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                loss, step_metrics = self._eval_step(inputs, targets)

                running_loss += loss * inputs.size(0)
                for name, metric_value in step_metrics.items():
                    metrics_result[name] += metric_value * inputs.size(0)

        val_loss = running_loss / total_samples
        val_metrics = {k: v / total_samples for k, v in metrics_result.items()}

        msg = 'Validation finished with loss:'
        print(f"{msg:<30} {val_loss: .10f} and metrics", end=" ")
        for k, v in val_metrics.items():
            print(f"{k}: {v: .10f}", end=", ")
        print()

        return val_loss, val_metrics

    def run(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = 10,
        patience: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs train and validation loops for the specified number of epochs or until early stopping.
        
        Args:
            train_loader (DataLoader): The DataLoader for the training data.
            val_loader (Optional[DataLoader]): The DataLoader for the validation data.
            num_epochs (Optional[int]): The number of epochs to train the model.
            patience (Optional[int]): The patience for early stopping.

        Returns:
            Dict[str, Any]: A dictionary containing the trained model and the training history.
        """
        if num_epochs is None and patience is None:
            raise ValueError("Must specify either number of epochs or patience and val_loader.")
        if num_epochs is None and val_loader is None:
            raise ValueError("If num_epochs is not specified, then val_loader must be specified for early stopping.")

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        no_improvement_count = 0
        history = {"train_loss": [], "train_metrics": [], "val_loss": [], "val_metrics": []}

        for epoch in range(num_epochs if num_epochs is not None else float('inf')):
            print(f'Starting training of epoch {epoch}:')
            train_loss, epoch_metrics = self.train_epoch(train_loader)

            history["train_loss"].append(train_loss)
            history["train_metrics"].append(epoch_metrics)

            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": epoch_metrics.get("mae", None),
                "train_rmse": epoch_metrics.get("rmse", None),
                "num_layers": self.model.lstm.num_layers if hasattr(self.model, "lstm") else None,
                "hidden_size": self.model.lstm.hidden_size if hasattr(self.model, "lstm") else None,
            }   

            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_metrics"].append(val_metrics)

                log_data.update({
                    "val_loss": val_loss,
                    "val_mae": val_metrics.get("mae", None),
                    "val_rmse": val_metrics.get("rmse", None),
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    optimzer_state = copy.deepcopy(self.optimizer.state_dict())
                    self._save_checkpoint(epoch, best_model_wts, optimzer_state, tag='best')

                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if patience is not None and no_improvement_count >= patience:
                    break
            else:
                if epoch % 10 == 0:
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    optimzer_state = copy.deepcopy(self.optimizer.state_dict())
                    self._save_checkpoint(epoch, best_model_wts, optimzer_state)
            wandb.log(log_data, step=epoch)

            print('-'*50)

        self.model.load_state_dict(best_model_wts)
        return {"model": self.model, "history": history}

    def _save_checkpoint(self, epoch: int, model_state_dict: Dict[str, Any], optimizer_state_dict: Dict[str, Any], tag: str='ckp') -> None:
        """
        Saves the model checkpoint.
        
        Args:
            epoch (int): The epoch number.
            model_state_dict (Dict[str, Any]): The model state dictionary.
            optimizer_state_dict (Dict[str, Any]): The optimizer state dictionary.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}_{tag}")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, f"checkpoint_{epoch}_{tag}.pt")
        torch.save(checkpoint, checkpoint_path)
    
    # TODO: Improve loading of checkpoint to better select folder
    def _load_checkpoint(self, checkpoint_file_path: str) -> None:
        """
        Loads the model checkpoint.
        
        Args:
            checkpoint_path (str): The path to the checkpoint file.
        
        Returns:
            Dict[str, Any]: The checkpoint dictionary.
        """
        if not checkpoint_file_path.endswith(".pt"):
            raise ValueError("Checkpoint file must be a .pt file.")
        if not os.path.exists(checkpoint_file_path):
            try:
                checkpoint_file_path = os.path.join(self.checkpoint_dir, checkpoint_file_path)
                if not os.path.exists(checkpoint_file_path):
                    raise FileNotFoundError
            except FileNotFoundError:
                print(f"Checkpoint file not found at: {checkpoint_file_path}")

        checkpoint = torch.load(checkpoint_file_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # TODO: Logging

    @property
    def checkpoint_dir(self) -> str:
        return self._checkpoint_dir
    
    @checkpoint_dir.setter
    def checkpoint_dir(self, value: str) -> None:
        if not os.path.exists(value):
            os.makedirs(value)
        self._checkpoint_dir = value

    @property
    def log_dir(self) -> str:
        return self._log_dir
    
    @log_dir.setter
    def log_dir(self, value: str) -> None:
        if not os.path.exists(value):
            os.makedirs(value)
        self._log_dir = value
