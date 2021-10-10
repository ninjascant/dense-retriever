import os
import json
import click
from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk, load_metric

from .models import BertDot

from .utils.gcs_utils import download_file_from_gcs
from .utils.file_utils import unzip_arch
from .preprocessing import construct_train_set, get_train_set_splits, tokenize_train_dataset, get_similar_docs
from .inference import InferenceRunner, save_inference_results, prepare_dataloader, prepare_dataset, extract_ids
from .ann_index import build_index_from_file


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
@click.option('-s', '--save-strategy', type=str, default='no')
def train_model(model_name, dataset_path, out_dir, batch_size, num_epochs, accum_steps, save_strategy):

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
        lr_scheduler_type='constant',
        learning_rate=3e-5,
        evaluation_strategy='epoch',
        save_strategy=save_strategy,
        output_dir='./tmp'
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.model.transformer.save_pretrained(out_dir)


class TestDataset(Dataset):
    def __init__(self, input_ids, attention_mask):

        self.input_ids = torch.tensor(input_ids)
        self.attention_mask = torch.tensor(attention_mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention': self.attention_mask[idx]
        }


def _construct_dataloader_from_encodings(encoding_file):
    with open(encoding_file) as json_file:
        encodings = json.load(json_file)

    ids = list(encodings.keys())
    input_ids = []
    attention_mask = []

    for doc_id in ids:
        input_ids.append(encodings[doc_id]['input_ids'])
        attention_mask.append(encodings[doc_id]['attention_mask'])

    dataset = TestDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    return dataloader, ids


def refresh_embeddings(dataset_dir, model_dir, out_dir, device, id_col):
    dataset = prepare_dataset(dataset_dir, torch_cols=['input_ids', 'attention_mask'])
    dataloader = prepare_dataloader(dataset, batch_size=32)
    ids = extract_ids(dataset, id_col)
    logger.info(f'Start inference; id_col: {id_col}')
    runner = InferenceRunner(model_name='bert-dot', model_dir_path=model_dir, device=device)
    logger.info(f'Finish inference; id_col: {id_col}')
    embeddings = runner.transform(dataloader, show_progress=True)
    save_inference_results(embeddings, ids, out_dir, True)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('model_name', type=str)
@click.argument('doc_dataset_dir', type=str)
@click.argument('query_dataset_dir', type=str)
@click.argument('out_dir', type=str)
@click.argument('refresh_steps', type=int)
@click.argument('total_steps', type=int)
@click.argument('encoding_file', type=str)
@click.argument('query_sample_file', type=str)
@click.option('-b', '--batch_size', type=int, default=8)
@click.option('-a', '--accum-steps', type=int, default=1)
@click.option('-t', '--top-n', type=int, default=50)
@click.option('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
def train_model_with_refresh(
    model_name,
    doc_dataset_dir,
    query_dataset_dir,
    out_dir,
    refresh_steps,
    total_steps,
    encoding_file,
    query_sample_file,
    batch_size,
    accum_steps,
    top_n,
    device
):
    train_args = TrainingArguments(
        max_steps=refresh_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=1000,
        lr_scheduler_type='constant',
        learning_rate=3e-5,
        evaluation_strategy='epoch',
        save_strategy='no',
        output_dir='./tmp'
    )

    refresh_iterations = int(total_steps / refresh_steps) + 1
    for i in range(refresh_iterations):
        logger.info(f'Iteration: {i+1}')
        if i == 0:
            continue
            # model = BertDot(model_name)
            # dataset = _load_dataset(dataset_path)
        else:
            model = BertDot(out_dir)

            logger.info('Updating doc embeddings')
            refresh_embeddings(
                dataset_dir=doc_dataset_dir,
                model_dir=out_dir,
                out_dir='model_outputs/doc_embeddings',
                device=device,
                id_col='doc_id'
            )

            logger.info('Updating query embeddings')
            refresh_embeddings(
                dataset_dir=query_dataset_dir,
                model_dir=out_dir,
                out_dir='model_outputs/query_embeddings',
                device=device,
                id_col='doc_id'
            )

            logger.info('Updating index')
            build_index_from_file('model_outputs/doc_embeddings', 'model_outputs/index.ann')

            logger.info('Constructing train set')
            logger.info('Finding similar docs')
            get_similar_docs(
                index_file='model_outputs/index.ann',
                query_embed_dir='model_outputs/query_embeddings',
                out_file='model_outputs/query_similar_docs.pkl',
                top_n=top_n
            )

            logger.info('Constructing train set')
            construct_train_set(
                doc_file=None,
                ann_res_file='model_outputs/query_similar_docs.pkl',
                query_sample_file=query_sample_file,
                out_file='model_outputs/train_set.json',
                top_n=top_n
            )

            get_train_set_splits(
                input_file='model_outputs/train_set.json',
                out_dir='model_outputs/train_set_refreshed',
                file_type='json',
            )

            logger.info('Tokenizing updated train set')
            tokenize_train_dataset(
                train_file_path='model_outputs/train_set_refreshed/train.csv',
                test_file_path='model_outputs/train_set_refreshed/test.csv',
                out_path='model_outputs/dataset_refreshed',
                model_name=model_name,
                file_type='csv',
                encodings_file=encoding_file
            )

            dataset = _load_dataset('model_outputs/dataset_refreshed')

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            compute_metrics=compute_metrics,
        )
        logger.info('Training model')
        trainer.train()
        logger.info('Saving intermediate results')
        trainer.model.transformer.save_pretrained(out_dir)
