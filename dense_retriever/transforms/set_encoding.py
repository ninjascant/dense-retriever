from typing import Any
from transformers import BertTokenizerFast
from datasets import load_dataset, Dataset
from .base import BaseTransform


def rename_encoded_column(dataset: Dataset, column_name: str) -> Dataset:
    dataset = dataset.rename_column('input_ids', f'{column_name}_input_ids')
    dataset = dataset.rename_column('attention_mask', f'{column_name}_attention_mask')
    dataset = dataset.remove_columns(['token_type_ids', column_name])
    return dataset


class TripletTokenizer(BaseTransform):
    def __init__(self, tokenizer_name_or_path: str):
        super(TripletTokenizer, self).__init__(transformer_out_path=None)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name_or_path)

    def _load_input_data(self, input_path: str):
        dataset = load_dataset('json', data_files=[input_path])['train']
        dataset = dataset.train_test_split(test_size=0.2)
        return dataset

    def _load_transformer(self, input_path: str):
        pass

    def _fit_transformer_fn(self, input_data: Any):
        pass

    def _transform_fn(self, input_data: Any):
        encoded_dataset = input_data.map(
            lambda example: self.tokenizer(example['query'], max_length=100, padding='max_length', truncation=True),
            batched=True,
            batch_size=10_000
        )
        encoded_dataset = rename_encoded_column(encoded_dataset, 'query')

        encoded_dataset = encoded_dataset.map(
            lambda example: self.tokenizer(example['pos_ctx'], max_length=512, padding='max_length', truncation=True),
            batched=True,
            batch_size=10_000
        )
        encoded_dataset = rename_encoded_column(encoded_dataset, 'pos_ctx')

        encoded_dataset = encoded_dataset.map(
            lambda example: self.tokenizer(example['neg_ctx'], max_length=512, padding='max_length', truncation=True),
            batched=True,
            batch_size=10_000
        )
        encoded_dataset = rename_encoded_column(encoded_dataset, 'neg_ctx')

        return encoded_dataset

    def _save_transformed_data(self, transformed_data: Dataset, out_path: str):
        transformed_data.save_to_disk(out_path)

    def _save_transformer(self, out_path: str):
        pass
