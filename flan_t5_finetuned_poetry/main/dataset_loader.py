import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class TextDataset(Dataset):
    """
    Textual dataset loader
    """

    def __init__(
            self,
            dataframe: DataFrame,
            tokenizer: T5Tokenizer,
            source_len: int,
            target_len: int,
            source_text: str,
            target_text: str
    ):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.target_text)

    def _get_tokenizer(self, data_text):
        return self.tokenizer.batch_encode_plus(
            [data_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data to ensure it's all strings
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self._get_tokenizer(source_text)
        target = self._get_tokenizer(target_text)

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long),
        }