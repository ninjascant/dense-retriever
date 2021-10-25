import os
import json
from typing import Union, Callable
import numpy as np
from transformers import Trainer
from datasets import load_from_disk
from .base import BaseEstimator


class BertDot(BaseEstimator):
    def __init__(
            self,
            model_name_or_path: str,
            model_type: str,
            train_steps: int,
            num_epochs: int,
            batch_size: int,
            accum_steps: int,
            lr: float = 3e-5,
            metric_fn: Callable = None,
            continue_train: bool = False,
            save_steps: int = None,
            in_batch_neg: bool = False,
    ):
        super(BertDot, self).__init__(
            model_name_or_path=model_name_or_path,
            model_type=model_type,
            train_steps=train_steps,
            num_epochs=num_epochs,
            batch_size=batch_size,
            accum_steps=accum_steps,
            save_steps=save_steps,
            lr=lr,
            continue_train=continue_train,
            metric_fn=metric_fn,
            in_batch_neg=in_batch_neg
        )

    def _load_model(self):
        model = self.model_class(self.model_name_or_path, in_batch_neg=self.in_batch_neg)
        return model

    def _save_model(self, trainer: Trainer, model_out_dir: str):
        trainer.model.transformer.save_pretrained(model_out_dir)

    def _load_dataset(self, dataset_path: str, torch_columns: Union[None, list]):
        dataset = load_from_disk(dataset_path)
        if torch_columns is None:
            dataset.set_format(type='torch')
        else:
            dataset.set_format(type='torch', columns=torch_columns)
        return dataset

    def _save_inference_results(self, inference_res, out_dir):
        embeddings, ids = inference_res
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings)
        with open(os.path.join(out_dir, 'ids.json'), 'w') as outfile:
            json.dump(ids, outfile, indent=2)
