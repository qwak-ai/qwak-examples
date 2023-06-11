import pandas as pd


def load_data(input_path: str, max_length=None):
    csv_df = pd.read_csv(input_path)
    if max_length:
        return csv_df.head(max_length)

    return csv_df
