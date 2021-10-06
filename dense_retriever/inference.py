import shutil
import os
import json
import click
from tqdm.auto import tqdm
from loguru import logger
import numpy as np
import torch
from datasets import load_from_disk
from .models import load_model
from .utils.file_utils import zip_dir


class InferenceRunner:
    def __init__(self, model_name, model_dir_path, device):
        self.max_len = 512
        self.batch_size = 32
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.tokenizer = None

        logger.info('Loading model')

        self.model = load_model(model_name, model_dir_path)
        self.model.to(self.device)

    def transform(self, dataloader, show_progress=False, print_progress_at=None):
        logger.info(f'Starting inference. Total steps: {len(dataloader)}')
        if show_progress:
            batch_iterator = tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            batch_iterator = enumerate(dataloader)

        self.model.eval()
        total_embeddings = []
        with torch.no_grad():

            for i, batch in batch_iterator:
                embeddings = self.model.get_embed(
                    batch['input_ids'].to(self.device),
                    batch['attention_mask'].to(self.device)
                )
                embeddings = embeddings.detach().cpu().numpy()
                total_embeddings.append(embeddings)
                if print_progress_at is not None and i != 0 and i % print_progress_at == 0:
                    logger.info(f'Iteration {i}')
        total_embeddings = np.vstack(total_embeddings)
        logger.info('Finished inference')
        return total_embeddings


def prepare_dataset(dataset_path, torch_cols):
    logger.info('Loading dataset')
    dataset = load_from_disk(dataset_path)
    dataset.set_format(type='torch', columns=torch_cols, output_all_columns=True)
    return dataset


def prepare_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)


def extract_ids(dataset, id_column):
    ids = dataset['test'][id_column]
    return ids


def remove_dir_if_exists(dir_path):
    try:
        shutil.rmtree(dir_path)
    except FileNotFoundError:
        pass


def save_inference_results(embeddings, ids, out_path, overwrite):
    if overwrite:
        remove_dir_if_exists(out_path)
        os.mkdir(out_path)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    np.save(os.path.join(out_path, 'embeddings.npy'), embeddings)
    with open(os.path.join(out_path, 'ids.json'), 'w') as outfile:
        json.dump(ids, outfile, indent=2)


def _run_inference(
        dataset_path,
        model_name,
        model_path,
        out_path,
        columns,
        batch_size,
        show_progress,
        print_progress_at,
        zip_path,
        device,
        overwrite,
        id_column
):
    columns = columns.split(',')

    inference_runner = InferenceRunner(model_name, model_path, device)

    dataset = prepare_dataset(dataset_path, columns)
    dataloader = prepare_dataloader(dataset, batch_size)

    embeddings = inference_runner.transform(dataloader, show_progress, print_progress_at)
    doc_ids = extract_ids(dataset, id_column)

    save_inference_results(embeddings, doc_ids, out_path, overwrite)

    if zip_path is not None:
        logger.info('Zipping dataset')
        zip_dir(out_path, zip_path)


@click.command()
@click.argument('dataset_path', type=str)
@click.argument('model_name', type=str)
@click.argument('model_path', type=str)
@click.argument('out_path', type=str)
@click.option('-c', '--columns', type=str, default='input_ids,attention_mask')
@click.option('-b', '--batch-size', type=int, default=32)
@click.option('-s', '--show-progress', is_flag=True)
@click.option('-p', '--print-progress-at', type=int, default=None)
@click.option('-z', '--zip-path', type=str, default=None)
@click.option('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
@click.option('-o', '--overwrite', is_flag=True)
@click.option('-i', '--id-column', type=str, default='doc_id')
def run_inference(
        dataset_path,
        model_name,
        model_path,
        out_path,
        columns,
        batch_size,
        show_progress,
        print_progress_at,
        zip_path,
        device,
        overwrite,
        id_column
):
    _run_inference(
        dataset_path,
        model_name,
        model_path,
        out_path,
        columns,
        batch_size,
        show_progress,
        print_progress_at,
        zip_path,
        device,
        overwrite,
        id_column
    )
