import logging
import click
from datasets import load_dataset, load_from_disk
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


@click.command()
@click.argument('file_path', type=str)
@click.argument('out_path', type=str)
@click.option('-f', '--file_type', type=str, default='csv')
@click.option('-d', '--delimiter', type=str, default=',')
@click.option('-c', '--column-names', default=None, type=str)
@click.option('-t', '--tokenizer-name', default='roberta', type=str)
@click.option('-z', '--zip-path', type=str, default=None)
def tokenize_dataset(file_path, out_path, file_type, delimiter, column_names, tokenizer_name, zip_path):
    logger.info('Loading tokenizer')
    tokenizer = init_tokenizer(tokenizer_name)

    logger.info('Loading dataset')
    dataset = load_dataset(file_type, data_files=file_path, column_names=column_names, delimiter=delimiter)

    logger.info('Tokenizing dataset')
    encoded_dataset = dataset.map(
        lambda example: tokenizer(example['text'], max_length=512, truncation=True, padding='max_length'),
        batched=True,
        batch_size=10_000,
    )
    encoded_dataset = encoded_dataset.remove_columns('text')
    logger.info('Saving dataset')
    encoded_dataset.save_to_disk(out_path)

    if zip_path is not None:
        logging.info('Zipping dataset')
        zip_dir(out_path, zip_path)
