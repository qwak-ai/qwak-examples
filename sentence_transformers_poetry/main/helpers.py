import torch


def get_device(pid):
    return (
        torch.device("cuda", pid % torch.cuda.device_count())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )