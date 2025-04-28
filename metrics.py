import torch

def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute the mean absolute error, between the true and predicted values."""
    return torch.mean(torch.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    """Compute the root mean squared error, between the true and predicted values."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def mean_absolute_percentage_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute the mean absolute percentage error, between the true and predicted values."""
    # Adding a small epsilon to avoid division by zero
    epsilon = torch.finfo(y_true.dtype).eps
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
