import os
import click
import numpy as np
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk, load_metric

from .models import BertDot

from .utils.gcs_utils import download_file_from_gcs
from .utils.file_utils import unzip_arch


def _download_dataset(bucket, gcs_path):
    local_arch_path = gcs_path.split('/')[-1]
    download_file_from_gcs(bucket, gcs_path, local_arch_path)
    unzip_arch(local_arch_path)
    os.remove(local_arch_path)
    dataset_dir = local_arch_path.split('.')[0]
    return dataset_dir


def _load_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)
    dataset.set_format(type='torch')
    return dataset


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def compute_metrics(eval_pred):
    metric = load_metric("f1")
    logits, labels = eval_pred
    predictions = softmax(logits)
    return metric.compute(predictions=predictions, references=labels)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('model_name', type=str)
@click.argument('dataset_path', type=str)
@click.argument('out_dir', type=str)
@click.option('-b', '--batch_size', type=int, default=8)
@click.option('-n', '--num_epochs', type=int, default=3)
@click.option('-a', '--accum-steps', type=int, default=1)
def train_model(model_name, dataset_path, out_dir, batch_size, num_epochs, accum_steps):

    dataset = _load_dataset(dataset_path)
    model = BertDot(model_name)
    train_args = TrainingArguments(
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=1000,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        output_dir=out_dir
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
