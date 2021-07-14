import sys
import os
import csv
import json
import logging
import click
from tqdm.auto import tqdm

csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def convert_tsv_to_ndjson(in_file_path, out_file_path, fieldnames, sample_size=None):
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
                outfile.write(json.dumps(line) + '\n')


@click.command()
@click.argument('input_file', type=str)
@click.argument('out_file', type=str)
@click.option('-f', '--field-names', type=str, default='doc_id,url,title,text')
@click.option('-s', '--sample-size', type=int, default=None)
def convert_msmarco(input_file, out_file, field_names, sample_size):
    field_names = field_names.split(',')
    logger.info(f'Converting {input_file} to {out_file}')
    convert_tsv_to_ndjson(input_file, out_file, field_names, sample_size)
