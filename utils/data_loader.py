import pickle
import pandas as pd

def load_data(data_path: str, filename: str):
    """
    Lädt Zeitreihendaten aus einem Pickle-File und splittet in Train/Val/Test.

    Args:
        data_path (str): Pfad zum Ordner mit der Pickle-Datei.
        filename (str): Dateiname der Pickle-Datei.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df
    """
    full_path = f"{data_path}/{filename}"
    with open(full_path, "rb") as f:
        pv_ts_dict = pickle.load(f)

    df = pv_ts_dict["data"].copy()

    # Zeitstempel konvertieren
    df["ts"] = pd.to_datetime(df["ts"], unit="ms") \
                  .dt.tz_localize("UTC") \
                  .dt.tz_convert("Europe/Berlin") \
                  .dt.tz_localize(None)

    # 6 Monate Test, davor 6 Monate Validierung
    latest_date = df["ts"].max()
    test_cutoff = latest_date - pd.DateOffset(months=6)
    val_cutoff = test_cutoff - pd.DateOffset(months=6)

    train_df = df[df["ts"] < val_cutoff]
    val_df   = df[(df["ts"] >= val_cutoff) & (df["ts"] < test_cutoff)]
    test_df  = df[df["ts"] >= test_cutoff]

    return train_df, val_df, test_df
