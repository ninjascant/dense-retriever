import logging
import os
from collections import OrderedDict
import configparser
import psycopg2
from psycopg2.extras import DictCursor
import click
from tqdm.auto import tqdm
from ..util import read_jsonl_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


class PSQLClient:
    def __init__(self, config_file_path):
        self.config = read_config(config_file_path)

    def make_ddl_query(self, query):
        conn = psycopg2.connect(**self.config)
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(query)
        conn.commit()
        conn.close()

    def make_dql_query(self, query):
        conn = psycopg2.connect(**self.config)
        with conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(query)
            res = cursor.fetchall()
        conn.close()
        return res

    def insert_values_from_dicts(self, rows, table_name):
        """
        Source: https://hakibenita.com/fast-load-data-python-postgresql
        """
        conn = psycopg2.connect(**self.config)
        with conn.cursor() as cursor:
            psycopg2.extras.execute_values(
                cursor,
                f'''INSERT INTO {table_name} VALUES %s;''',
                (tuple(row.values()) for row in rows))
        conn.commit()
        conn.close()


def read_config(config_file_path):
    parser = configparser.ConfigParser()
    parser.read(config_file_path)
    config = parser.items('postgresql')
    config = dict(config)
    return config


def get_msmarco_doc_table_query(table_name):
    return f'''
        CREATE TABLE {table_name}(
            doc_id TEXT,
            url TEXT,
            title TEXT,
            text TEXT,
            idx SERIAL PRIMARY KEY
        );
    '''


def create_msmarco_row(sample):
    return OrderedDict(doc_id=sample['doc_id'], url=sample['url'], title=sample['title'], text=sample['text'])


def upload_msmarco_file(file_path, db_config_path, table_name):
    data = read_jsonl_file(file_path)
    data = [create_msmarco_row(row) for row in data]
    client = PSQLClient(db_config_path)
    client.insert_values_from_dicts(data, table_name)


def upload_msmarco(input_dir, db_config_path, table_name, offset):
    logger.info(f'Started uploading data to {table_name}')
    files = os.listdir(input_dir)
    files = files[offset:]
    for file in tqdm(files):
        upload_msmarco_file(os.path.join(input_dir, file), db_config_path, table_name)


@click.command()
@click.argument('dataset_type', type=str)
@click.argument('config_file_path', type=str)
@click.argument('table_name', type=str)
def create_table(dataset_type, config_file_path, table_name):
    if dataset_type == 'msmarco':
        query = get_msmarco_doc_table_query(table_name)
    else:
        raise NotImplementedError('Only MSMARCO dataset type is currently supported')

    client = PSQLClient(config_file_path)
    logger.info(f'Creating table {table_name} with query\n{query}')
    client.make_ddl_query(query)


@click.command()
@click.argument('dataset_type', type=str)
@click.argument('config_file_path', type=str)
@click.argument('table_name', type=str)
@click.argument('input_dir', type=str)
@click.option('-o', '--offset', type=int, default=0)
def upload_data(dataset_type, config_file_path, table_name, input_dir, offset):
    if dataset_type == 'msmarco':
        upload_msmarco(input_dir, config_file_path, table_name, offset)
    else:
        raise NotImplementedError('Only MSMARCO dataset type is currently supported')
