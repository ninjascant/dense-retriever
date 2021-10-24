from typing import Any

from datasets import load_from_disk
from .base import BaseTransform
from ..utils.redis_utils import RedisClient


class EncodingCacheBuilder(BaseTransform):
    def __init__(self):
        super(EncodingCacheBuilder, self).__init__(transformer_out_path=None)

        self.client = RedisClient(hostname='localhost')

    def _load_input_data(self, input_path: str):
        dataset = load_from_disk(input_path)
        return dataset

    def _transform_fn(self, input_data: Any):
        input_data.map(lambda row:
                       self.client.write(
                           row['id'],
                           {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']}
                       ))

    def _save_transformed_data(self, transformed_data: Any, out_path: str):
        pass

    def _save_transformer(self, out_path: str):
        pass

    def _load_transformer(self, input_path: str):
        pass

    def _fit_transformer_fn(self, input_data: Any):
        pass
