import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from .utils.file_utils import zip_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def init_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def _rename_torch_columns(dataset, colunm_name):
    dataset = dataset.rename_column('input_ids', f'{colunm_name}_input_ids')
    dataset = dataset.rename_column('attention_mask', f'{colunm_name}_attention_mask')
    dataset = dataset.rename_column('token_type_ids', f'{colunm_name}_token_type_ids')
    return dataset


def _encode_text_column(dataset, tokenizer, column_name, max_length, padding):
    encoded_dataset = dataset.map(
        lambda example: tokenizer(example[column_name], max_length=max_length, padding=padding, truncation=True),
        batched=True,
        batch_size=10_000
    )
    encoded_dataset = encoded_dataset.remove_columns(column_name)
    encoded_dataset = _rename_torch_columns(encoded_dataset, column_name)
    return encoded_dataset


def tokenize_dataset(
        train_file_path,
        test_file_path,
        out_path,
        model_name,
        zip_path,
        max_length,
        padding
):
    logger.info('Loading dataset')
    if test_file_path is None:
        dataset = load_dataset('json', data_files=train_file_path)
    else:
        dataset = load_dataset('json', data_files={'train': train_file_path, 'test': test_file_path})

    tokenizer = init_tokenizer(model_name)

    logger.info('Tokenizing dataset')
    encoded_dataset = _encode_text_column(dataset, tokenizer, 'query', max_length, padding)
    encoded_dataset = _encode_text_column(encoded_dataset, tokenizer, 'doc', max_length, padding)

    logger.info('Saving dataset')
    encoded_dataset.save_to_disk(out_path)

    if zip_path is not None:
        logging.info('Zipping dataset')
        zip_dir(out_path, zip_path)
