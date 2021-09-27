import sys
import os
import csv
import json
import logging
import click
from tqdm.auto import tqdm
from ..utils.file_utils import read_qrel_file

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def truncate_text(sample, max_len):
    sample['text'] = sample['text'][:max_len]
    return sample


def extract_sample_text(sample):
    if sample['text'] is None:
        sample['text'] = ''
    text = sample['url'] + '<sep>' + sample['title'] + '<sep>' + sample['text']
    return {'doc_id': sample['doc_id'], 'text': text}


def convert_tsv_to_ndjson(in_file_path, out_file_path, fieldnames, sample_size=None, max_len=None, filter_func=None):
    if os.path.exists(out_file_path):
        os.remove(out_file_path)
    with open(in_file_path) as csv_file:
        reader = csv.DictReader(
            csv_file,
            dialect='excel-tab',
            fieldnames=fieldnames
        )
        with open(out_file_path, 'a') as outfile:
            for i, line in tqdm(enumerate(reader)):
                if sample_size is not None and i > sample_size:
                    break
                if filter_func is not None and not filter_func(line):
                    continue
                converted_sample = extract_sample_text(line)
                if max_len is not None:
                    converted_sample = truncate_text(converted_sample, max_len)
                outfile.write(json.dumps(converted_sample) + '\n')


@click.command()
@click.argument('input_file', type=str)
@click.argument('out_file', type=str)
@click.option('-f', '--field-names', type=str, default='doc_id,url,title,text')
@click.option('-s', '--sample-size', type=int, default=None)
@click.option('-m', '--max-len', type=int, default=None)
def convert_msmarco(input_file, out_file, field_names, sample_size, max_len):
    field_names = field_names.split(',')
    logger.info(f'Converting {input_file} to {out_file}')
    convert_tsv_to_ndjson(input_file, out_file, field_names, sample_size, max_len=max_len)


@click.command()
@click.argument('doc_file', type=str)
@click.argument('qrel_file', type=str)
@click.argument('out_file', type=str)
@click.option('-f', '--field-names', type=str, default='doc_id,url,title,text')
@click.option('-s', '--sample-size', type=int, default=None)
@click.option('-m', '--max-len', type=int, default=None)
def extract_train_set(doc_file, qrel_file, out_file, field_names, sample_size, max_len):
    field_names = field_names.split(',')

    qrels = read_qrel_file(qrel_file)
    train_doc_ids = set(qrels['doc_id'].values)

    def filter_train(row):
        if row['doc_id'] in train_doc_ids:
            return True
        else:
            return False

    convert_tsv_to_ndjson(
        doc_file,
        out_file,
        field_names,
        max_len=max_len,
        sample_size=sample_size,
        filter_func=filter_train)
