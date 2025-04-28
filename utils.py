import os
import json
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torchvision as tv


def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """
    Save processed data to a file.
    """
    if df is None:
        raise ValueError("Processed data is empty.")
    
    if not path.endswith('.json'):
        raise ValueError("File path must have a .json extension.")
    
    if not os.path.exists(os.path.dirname(path)):
        raise FileNotFoundError(f"Directory {os.path.dirname(path)} does not exist.")

    # Convert datetime columns to ISO 8601 string format
    df['start'] = df['start'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    df['end'] = df['end'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')

    print(f'{df.start.head()}, {df.end.head()}')
    # Save to JSON
    df.to_json(path, orient='records', date_format='iso', date_unit='s', index=False)
    print(f"Processed data saved to {path}")


def hour_15_min_interval(dt: pd.Timestamp) -> str:
    """
    Convert a datetime to a string in the format HH:MM representing a 15-minute interval.
    """
    hour = dt.hour
    # Calculate 15-minute interval (0 for minutes 0-14, 1 for 15-29, etc.)
    interval = (dt.minute // 15) * 15

    return f"{hour:02d}:{interval:02d}"


def sample_poisson_arrival(probs: np.ndarray, rows: int, scaler: int=10) -> np.ndarray:
    """
    Sample n Poisson arrivals from a Poisson distribution. For an expected number of events.
    Calculation is: lambda = probs * scaler. Where probs is the probability of arrival for each interval.
    And scaler is the expected number of events.

    Parameters
    ----------
    probs : np.ndarray
        Probabilities of arrival for each interval. Sum up to 1.
    rows : int
        Size of the output array.
    scaler : tuple
        Scaler to adjust the mean number of events. Default is 10.

    Returns
    -------
    np.ndarray
        Sampled Poisson arrivals. Of shape (range, len(probs)).
    """

    lambda_values = probs * scaler
    return np.random.poisson(lam=lambda_values, size=(rows, len(lambda_values)))


def flatten_df(df):
    for col in df.columns:
        if col in df.columns and isinstance(df[col].iloc[0], list):
            s = df[col].apply(pd.Series)
            s.columns = [col + '_' + str(c) for c in s.columns]
            df = pd.concat([df, flatten_df(pd.DataFrame(s))], axis=1).drop(columns=[col])
        if col in df.columns and isinstance(df[col].iloc[0], dict):
            s = df[col].apply(pd.Series)
            s.columns = [col + '_' + str(c) for c in s.columns]
            df = pd.concat([df, flatten_df(pd.DataFrame(s))], axis=1).drop(columns=[col])
    return df


def create_dataset(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i: i + lookback])
        y.append(data[i + lookback: i + lookback + horizon])
    return t.tensor(X).float().unsqueeze(-1), t.tensor(y).float()


def create_sequences(dataframe, seq_length, horizon, target_col):
    v_index = dataframe.columns.get_loc(target_col)
    print(f'target col idx: {v_index}')

    sequences = []
    targets = []
    data = dataframe.values
    if not len(data.shape) == 2:
        data.unsqueeze(-1)
    for i in range(len(data) - seq_length - horizon + 1):
        seq = data[i : i + seq_length, :]  # Features
        target = data[i + seq_length : i + seq_length + horizon, v_index]  # Target horizon
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def train(model,
          X_train,
          y_train,
          X_val,
          y_val,
          epochs,
          batch_size,
          patience=5,
          lr=0.001,
          name='best_model.pt',
          device='cuda',
          *args,
          **kwargs):
    criterion = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    model.train()

    base_lr =  lr / 100

    scheduler = t.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=base_lr, 
        max_lr=lr, 
        step_size_up=len(X_train) // batch_size, 
        mode='triangular2'
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training loop
        model.train()
        for i in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            X_batch = X_train[i: i + batch_size].to(device)
            y_batch = y_train[i: i + batch_size].to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with t.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val[i: i + batch_size].to(device)
                y_batch = y_val[i: i + batch_size].to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        val_loss /= (len(X_val) / batch_size)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            t.save(model.state_dict(), name)  # Save the best model
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Stopping early at epoch {epoch}')
            break

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Training loss: {loss.item()}, Validation loss: {val_loss}')

    # Return the best model's state_dict
    model.load_state_dict(t.load(name))
    return model


def evaluate(model, test_dataloader, device='cuda'):
    model.eval()
    model = model.to(device)
    criterion = nn.MSELoss()
    total_loss = 0
    all_predictions = []

    with t.no_grad():
        # Process in batches
        for X_batch, y_batch in test_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            y_pred = model(X_batch)
            
            all_predictions.append(y_pred.detach().cpu())

    avg_loss = total_loss / len(test_dataloader)
    all_predictions = t.cat(all_predictions, dim=0)  # Concatenate all batches

    return avg_loss, all_predictions


def train_epochs(model, train_dataloader, epochs, batch_size, lr=0.0001, name='best_model.pt'):
    criterion = nn.MSELoss()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    model = model.to('cuda')
    model.train()

    for epoch in range(epochs):
        # Training loop
        model.train()
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            X_batch = X_batch.to('cuda')
            y_batch = y_batch.to('cuda')

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

    t.save(model.state_dict(), name)
    return model