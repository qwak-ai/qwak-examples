import os
from urllib.parse import urlparse

import pandas as pd
import torch
from pandas import DataFrame


def get_device():
    pid = os.getpid()
    return (
        torch.device("cuda", pid % torch.cuda.device_count())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def is_valid_uri(input_string):
    try:
        result = urlparse(input_string)
        return all([result.scheme, result.netloc])
    except (ValueError, TypeError):
        return False


def get_local_path(file_name: str) -> str:
    """
    Returns the absolute path of a local file
    """
    running_file_absolute_path = os.path.dirname(
        os.path.abspath(__file__)
    )
    return os.path.join(
        running_file_absolute_path,
        file_name
    )


def load_data(input_path: str = None, max_length: int = None) -> DataFrame:
    """
    Load data from a CSV in either a remote or local path
    """
    if is_valid_uri(input_path):
        file_path = input_path
    else:
        file_path = get_local_path(input_path)

    csv_df = pd.read_csv(file_path)
    if max_length and max_length > 0:
        return csv_df.head(max_length)

    return csv_df


def write_data(output_path: str, df):
    """
    Saving data to a local CSV file
    """
    df.to_csv(output_path, index=False)