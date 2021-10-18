from typing import Union
from tqdm.auto import tqdm
from loguru import logger
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_metric


def extract_ids(dataset, id_column):
    ids = dataset['test'][id_column]
    return ids


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def compute_f1(eval_pred):
    metric = load_metric("f1")
    logits, labels = eval_pred
    predictions = softmax(logits)
    return metric.compute(predictions=predictions, references=labels)


class BaseEstimator:
    def __init__(
            self,
            model_name_or_path: str,
            train_steps: int,
            num_epochs: int,
            batch_size: int,
            accum_steps: int,
            lr: float = 3e-5,
            metric_fn=compute_f1,
            device=None,
            eval_batch_size=None
    ):
        self.model_name_or_path = model_name_or_path
        self.train_steps = train_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.lr = lr

        self.model = self._load_model()
        self.metric_fn = metric_fn

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if eval_batch_size is None:
            self._eval_batch_size = batch_size
        else:
            self._eval_batch_size = eval_batch_size

    def _load_model(self):
        raise NotImplementedError('You need to implement this method')

    def _load_dataset(self, dataset_path: str, torch_columns: Union[None, list]):
        raise NotImplementedError('You need to implement this method')

    def _save_model(self, trainer: Trainer, model_out_dir: str):
        raise NotImplementedError('You need to implement this method')

    def _save_inference_results(self, inference_res, out_dir):
        raise NotImplementedError('You need to implement this method')

    def fit(self, dataset_dir: str, model_out_dir: str):
        dataset = self._load_dataset(dataset_dir, torch_columns=None)
        train_args = TrainingArguments(
            max_steps=self.train_steps,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.accum_steps,
            learning_rate=self.lr,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=1000,
            lr_scheduler_type='constant',
            evaluation_strategy='no',
            save_strategy='no',
            output_dir='./tmp'
        )
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            compute_metrics=self.metric_fn
        )
        logger.info('Starting training')
        trainer.train()
        logger.info('Finished training')

        self._save_model(trainer, model_out_dir)

    def predict(self, dataset_dir, out_dir, id_col='doc_id'):
        logger.info('Loading data')
        dataset = self._load_dataset(dataset_dir, torch_columns=['input_ids', 'attention_mask'])
        dataloader = torch.utils.data.DataLoader(dataset['test'], self._eval_batch_size, shuffle=False)
        batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader))
        ids = extract_ids(dataset, id_col)

        self.model.to(self.device)
        self.model.eval()
        total_embeddings = []
        logger.info('Starting inference')
        with torch.no_grad():
            for i, batch in batch_iterator:
                embeddings = self.model.get_embed(
                    batch['input_ids'].to(self.device),
                    batch['attention_mask'].to(self.device)
                )
                embeddings = embeddings.detach().cpu().numpy()
                total_embeddings.append(embeddings)
        total_embeddings = np.vstack(total_embeddings)
        logger.info('Finished inference')

        self._save_inference_results((total_embeddings, ids), out_dir)

        return total_embeddings
