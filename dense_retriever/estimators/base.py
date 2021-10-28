from typing import Union
from tqdm.auto import tqdm
from loguru import logger
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_metric
from ..models.bert_dot import BertDotBCEModel, BertDotPairwiseRankingModel, BertEmbedModel, BertClsModel

MODEL_TYPES = {
    'bert-dot-bce': BertDotBCEModel,
    'bert-dot-pairwise-ranking': BertDotPairwiseRankingModel,
    'bert-linear-cls': BertClsModel
}


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


class IRTrainer(Trainer):
    @staticmethod
    def prediction_step(model, inputs, prediction_loss_only, ignore_keys):
        with torch.no_grad():
            output = model(inputs['input_ids'].to('cuda'), inputs['attention_mask'].to('cuda'))
            logits = output[1].detach().cpu().numpy()
        return None, logits, None


class BaseEstimator:
    def __init__(
            self,
            model_name_or_path: str,
            model_type: str,
            train_steps: int,
            num_epochs: int,
            batch_size: int,
            accum_steps: int,
            lr: float = 3e-5,
            metric_fn=compute_f1,
            device=None,
            eval_batch_size=None,
            save_steps=None,
            continue_train=False,
            in_batch_neg=False
    ):
        self.model_name_or_path = model_name_or_path
        self.model_class = MODEL_TYPES[model_type]
        self.train_steps = train_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.lr = lr

        self.in_batch_neg = in_batch_neg
        self.metric_fn = metric_fn

        self.model = self._load_model()

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if eval_batch_size is None:
            self._eval_batch_size = batch_size
        else:
            self._eval_batch_size = eval_batch_size

        if save_steps is None:
            self._save_strategy = 'epoch'
            self._save_steps = 0
        else:
            self._save_strategy = 'steps'
            self._save_steps = save_steps
        self._continue_train = continue_train

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
            logging_steps=500,
            lr_scheduler_type='constant',
            evaluation_strategy='steps',
            eval_steps=5000,
            save_strategy=self._save_strategy,
            save_steps=self._save_steps,
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
        trainer.train(resume_from_checkpoint=self._continue_train)
        logger.info('Finished training')
        trainer.save_state()
        self._save_model(trainer, model_out_dir)

    def predict(self, dataset_dir, out_dir, id_col='doc_id'):
        logger.info('Loading data')

        dataset = self._load_dataset(dataset_dir, torch_columns=['input_ids', 'attention_mask'])
        model = BertEmbedModel(self.model_name_or_path)
        trainer = IRTrainer(
            model=model,
        )
        predictions = trainer.predict(test_dataset=dataset['test'])
        predictions = predictions.predictions
        ids = dataset['test'][id_col]
        self._save_inference_results((predictions, ids), out_dir)
