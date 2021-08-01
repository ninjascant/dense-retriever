import os
from pathlib import Path
import logging
import click
from datasets import load_dataset
from transformers import RobertaTokenizerFast
from .util import zip_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)

TOKENIZERS = {
    'roberta': {
        'class': RobertaTokenizerFast,
        'model_name': 'roberta-base'
    }
}


def init_tokenizer(tokenizer_name):
    tokenizer = TOKENIZERS.get(tokenizer_name)['class'].from_pretrained(TOKENIZERS.get(tokenizer_name)['model_name'])
    return tokenizer


def tokenize_file(file_path, file_type, out_path, zip_path, tokenizer, column_names, delimiter, padding):
    logger.info('Loading dataset')
    if file_type == 'csv':
        dataset = load_dataset(file_type, data_files=file_path, column_names=column_names, delimiter=delimiter)
    else:
        dataset = load_dataset(file_type, data_files=file_path)

    logger.info('Tokenizing dataset')
    encoded_dataset = dataset.map(
        lambda example: tokenizer(example['text'], max_length=512, truncation=True, padding=padding),
        batched=True,
        batch_size=10_000,
    )
    encoded_dataset = encoded_dataset.remove_columns('text')
    logger.info('Saving dataset')
    encoded_dataset.save_to_disk(out_path)

    if zip_path is not None:
        logging.info('Zipping dataset')
        zip_dir(out_path, zip_path)


def tokenize_dir(dir_path, out_path, zip_path, tokenizer, file_type, column_names, delimiter, padding):
    files = sorted(os.listdir(dir_path))
    for file in files:
        file_name = Path(file).stem
        file_input_path = os.path.join(dir_path, file)
        file_out_path = os.path.join(out_path, file_name)
        file_zip_path = os.path.join(zip_path, file_name + '.tar.gz')
        tokenize_file(
            file_input_path,
            file_type,
            file_out_path,
            file_zip_path,
            tokenizer,
            column_names,
            delimiter,
            padding
        )


@click.command()
@click.argument('input_path', type=str)
@click.argument('out_path', type=str)
@click.option('-f', '--file_type', type=str, default='csv')
@click.option('-d', '--delimiter', type=str, default=',')
@click.option('-c', '--column-names', default=None, type=str)
@click.option('-t', '--tokenizer-name', default='roberta', type=str)
@click.option('-z', '--zip-path', type=str, default=None)
@click.option('-p', '--padding', type=str, default='max_length')
def tokenize_dataset(input_path, out_path, file_type, delimiter, column_names, tokenizer_name, zip_path, padding):
    logger.info('Loading tokenizer')
    tokenizer = init_tokenizer(tokenizer_name)

    if os.path.isdir(input_path):
        tokenize_dir(input_path, out_path, zip_path, tokenizer, file_type, column_names, delimiter, padding)
    else:
        tokenize_file(input_path, file_type, out_path, zip_path, tokenizer, column_names, delimiter, padding)
