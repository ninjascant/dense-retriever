import os
import logging
import click
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def read_text_file(file_path):
    with open(file_path) as file:
        data = file.readlines()
    return data


def concat_text(row):
    if type(row['title']) is not str and type(row['abstract']) is str:
        return row['abstract']
    elif type(row['title']) is str and type(row['abstract']) is not str:
        return row['title']
    else:
        return row['title'] + '<sep>' + row['abstract']


@click.command()
@click.argument('data_dir', type=str)
@click.option('-o', '--out-file-name', type=str, default='text_dataset.csv')
@click.option('-s', '--sample-size', type=int, default=None)
def get_trec_covid_abstracts(data_dir, out_file_name, sample_size):
    doc_ids = read_text_file(os.path.join(data_dir, 'docids-rnd5.txt'))
    doc_ids = [row.replace('\n', '') for row in doc_ids]

    metadata = pd.read_csv(os.path.join(data_dir, '2021-07-12', 'metadata.csv'))

    text_dataset = metadata.loc[
        (~metadata['title'].isna() | ~metadata['abstract'].isna()) &
        metadata['cord_uid'].isin(set(doc_ids))]
    if sample_size is not None:
        text_dataset = metadata.sample(sample_size)
    text_dataset['text'] = text_dataset.apply(concat_text, axis=1)
    text_dataset = text_dataset[['text', 'cord_uid']]
    text_dataset = text_dataset.reset_index(drop=True)

    out_path = os.path.join(data_dir, out_file_name)
    logger.info(f'Saving text dataset to {out_path}')
    text_dataset.to_csv(out_path, index_label='idx')
