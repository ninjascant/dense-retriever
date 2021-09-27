import os
import json
import logging
from tqdm.auto import tqdm
import click
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def load_json_file(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


def load_embeddings_to_index(input_dir, index):
    subdirs = [item for item in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, item))]
    num_embed_subdirs = len(subdirs)
    for i in tqdm(range(num_embed_subdirs)):
        embeddings = np.load(os.path.join(input_dir, f'embeddings_{i}', 'embeddings.npy')).astype(np.float32)
        ids = load_json_file(os.path.join(input_dir, f'embeddings_{i}', 'ids.json'))
        ids = [int(doc_id[1:]) for doc_id in ids]
        ids = np.array(ids).astype(np.int64)
        index.add_with_ids(embeddings, ids)
    return index


@click.command()
@click.argument('input_dir', type=str)
@click.argument('out_dir', type=str)
def build_index(input_dir, out_dir):
    import faiss
    logger.info('Initializing index')
    index = faiss.index_factory(312, 'IDMap,Flat', faiss.METRIC_INNER_PRODUCT)
    logger.info('Adding embeddings to index')
    load_embeddings_to_index(input_dir, index)
    logger.info('Saving index')
    faiss.write_index(index, os.path.join(out_dir, 'index.ann'))


def load_index(index_path):
    import faiss
    logger.info('Loading index')
    index = faiss.read_index(index_path)
    logger.info('Loaded index')
    return index


def load_query_embeddings(query_embed_path):
    query_embeds = np.load(query_embed_path).astype(np.float32)
    return query_embeds


def run_batch_search(index_path, query_embed_path, out_path, top_n=100):
    index = load_index(index_path)
    query_embeds = load_query_embeddings(query_embed_path)

    logger.info('Starting search')
    _, res_idx = index.search(query_embeds, 100)
    logger.info('Finished search')
    np.save(out_path, res_idx)
    return res_idx


@click.command()
@click.argument('index_path', type=str)
@click.argument('query_embed_path', type=str)
@click.argument('qrel_path', type=str)
@click.argument('out_path', type=str)
def validate(index_path, query_embed_path, qrel_path, out_path):
    res_ids = run_batch_search(index_path, query_embed_path, out_path)

    qrels = pd.read_csv(qrel_path, sep=' ', header=None)
    qrels.columns = ['qid', 'none1', 'doc_id', 'none2']
    qrels = qrels[['qid', 'doc_id']]

    queries = pd.read_csv('../ir_datasets/msmarco_docs/raw/dev-queries.tsv', sep='\t', header=None)
    queries.columns = ['qid', 'text']

    logger.info('Calculating MRR@100')
    ranks = []
    for i, res in enumerate(res_ids):
        res = [item for item in res]
        qid = queries.iloc[i]['qid']
        true_value = int(qrels.loc[qrels['qid'] == qid]['doc_id'].values[0][1:])
        if true_value in res:
            rank = 1 / (res.index(true_value) + 1)
        else:
            rank = 0
        ranks.append(rank)
    mrr = np.mean(ranks)
    print(mrr)


@click.command()
@click.argument('index_path', type=str)
@click.argument('query_embed_path', type=str)
@click.argument('out_path', type=str)
@click.option('-t', '--top-n', type=int, default=20)
def get_train_samples(index_path, query_embed_path, out_path, top_n):
    run_batch_search(index_path, query_embed_path, out_path, top_n=top_n)
