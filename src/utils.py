import pandas as pd


def read_logs(path):
    """
    Read metrics.csv produced by pytorch_lightning.loggers.CSVLogger into a dictionary.
    """

    logs = pd.read_csv(path)
    return {c: logs[c].dropna().reset_index(drop=True) for c in logs.columns}
