import pandas as pd
import torch


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(input_path: str, max_length=None):
    csv_df = pd.read_csv(input_path)

    if max_length and max_length > 0:
        return csv_df.head(max_length)

    return csv_df
