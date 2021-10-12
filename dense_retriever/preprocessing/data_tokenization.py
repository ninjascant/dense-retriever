import os
import json
from tqdm.auto import tqdm
from loguru import logger
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, BertTokenizerFast
from ..utils.file_utils import zip_dir, write_pickle_file
from ..utils.gcs_utils import download_file_from_gcs
from ..utils.redis_utils import RedisClient


def init_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def _rename_torch_columns(dataset, column_name):
    dataset = dataset.rename_column('input_ids', f'{column_name}_input_ids')
    dataset = dataset.rename_column('attention_mask', f'{column_name}_attention_mask')
    # dataset = dataset.rename_column('token_type_ids', f'{column_name}_token_type_ids')
    return dataset


def _encode_text_column(dataset, tokenizer, column_name, max_length, padding, rename_cols=True):
    encoded_dataset = dataset.map(
        lambda example: tokenizer(example[column_name], max_length=max_length, padding=padding, truncation=True),
        batched=True,
        batch_size=10_000
    )
    encoded_dataset = encoded_dataset.remove_columns(column_name)
    if rename_cols:
        encoded_dataset = _rename_torch_columns(encoded_dataset, column_name)
    return encoded_dataset


def _set_encoding_from_cache(dataset, column_name, rename_cols=True):
    client = RedisClient(hostname='localhost')

    def get_encoding(example):
        doc_id = example['doc']
        encoding = client.read(doc_id)
        return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask']}

    encoded_dataset = dataset.map(get_encoding)
    encoded_dataset = encoded_dataset.remove_columns(column_name)
    if rename_cols:
        encoded_dataset = _rename_torch_columns(encoded_dataset, column_name)
    return encoded_dataset


def tokenize_train_dataset(
        train_file_path,
        test_file_path,
        out_path,
        model_name,
        file_type='csv',
        zip_path=None,
        max_length=512,
        padding='max_length',
        use_cache=False
):
    logger.info('Loading dataset')
    if test_file_path is None:
        dataset = load_dataset(file_type, data_files=train_file_path)
    else:
        dataset = load_dataset(file_type, data_files={'train': train_file_path, 'test': test_file_path})

    tokenizer = init_tokenizer(model_name)

    if not use_cache:
        logger.info('Tokenizing dataset')
        encoded_dataset = _encode_text_column(dataset, tokenizer, 'query', max_length, padding)
        encoded_dataset = _encode_text_column(encoded_dataset, tokenizer, 'doc', max_length, padding)
    else:
        encoded_dataset = _set_encoding_from_cache(dataset, 'query', True)
        encoded_dataset = _set_encoding_from_cache(encoded_dataset, 'doc', True)

    logger.info('Saving dataset')
    encoded_dataset.save_to_disk(out_path)

    if zip_path is not None:
        logger.info('Zipping dataset')
        zip_dir(out_path, zip_path)


def tokenize_test_dataset(
        input_file,
        out_dir,
        model_name,
        file_type,
        zip_path,
        max_length,
        padding
):
    logger.info('Loading dataset')
    dataset = load_dataset(file_type, data_files={'test': input_file})
    tokenizer = init_tokenizer(model_name)

    logger.info('Tokenizing dataset')
    encoded_dataset = _encode_text_column(dataset, tokenizer, 'text', max_length, padding,
                                          rename_cols=False)
    print(encoded_dataset.column_names)

    logger.info('Saving dataset')
    encoded_dataset.save_to_disk(out_dir)

    if zip_path is not None:
        logger.info('Zipping dataset')
        zip_dir(out_dir, zip_path)


def truncate_text(text, max_words):
    return ' '.join(text.split()[:max_words])


def truncate_docs(
    input_file,
    out_file
):
    with open(out_file, 'a') as outfile:
        with open(input_file) as file:
            for line in tqdm(file.readlines()):
                line_dict = json.loads(line)
                line_dict['text'] = truncate_text(line_dict['text'], 550)
                outfile.write(json.dumps(line_dict) + '\n')


def prepare_encoding_cache(input_file, out_file):
    with open(input_file) as file:
        docs = [json.loads(line) for i, line in enumerate(file.readlines())]

    tokenizer = BertTokenizerFast.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

    texts = [row['text'] for row in docs]
    logger.info('Start tokenizing')
    encodings = tokenizer.batch_encode_plus(texts, max_length=512, padding='max_length', truncation=True)
    logger.info('End tokenizing')

    encodings = [{
        'doc_id': row['doc_id'],
        'encodings': {'input_ids': encodings['input_ids'][i], 'attention_mask': encodings['attention_mask'][i]}
    } for i, row in enumerate(docs)]

    write_pickle_file(out_file, encodings)


def export_encoding_to_redis(encoding_dataset_dir):
    client = RedisClient(hostname='localhost')
    dataset = load_from_disk(encoding_dataset_dir)
    dataset.map(lambda row:
                client.write(row['doc_id'], {'input_ids': row['input_ids'], 'attention_mask': row['attention_mask']}))

