import os
import itertools
import json
from loguru import logger
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..data_model import QuerySample, ANNSearchRes, TrainSampleData, IRTrainSample, IRTrainSampleWithoutDoc
from ..utils.file_utils import read_pickle_file, write_pickle_file, write_jsonl_file
from ..ann_index import load_index


def truncate_text(text, max_words):
    return ' '.join(text.split()[:max_words])


def indices_to_ids(indices):
    indices = [idx for idx in indices if idx != -1]
    return ['D' + str(idx) for idx in indices]


def get_similar_docs(index_file, query_embed_dir, out_file, top_n):
    query_embeddings = np.load(os.path.join(query_embed_dir, 'embeddings.npy'))
    with open(os.path.join(query_embed_dir, 'ids.json')) as json_file:
        query_ids = json.load(json_file)
    index = load_index(index_file)
    logger.info('Starting search')
    _, res_indices = index.search(query_embeddings, top_n)
    logger.info('Finished search')
    res_ids = [indices_to_ids(indices) for indices in res_indices]

    ann_results = [ANNSearchRes(query_id=query_id, search_results=res_ids[i]) for i, query_id in enumerate(query_ids)]
    write_pickle_file(out_file, ann_results)


def construct_ir_sample(sample_data, docs=None, top_n=100, max_words=550):
    hard_negative_idx = np.random.randint(low=-1, high=top_n-1)
    hard_negative_id = sample_data.similar_doc_ids[hard_negative_idx]

    if docs is not None:
        hard_negative_doc = docs.loc[hard_negative_id]['text']
        positive_doc = docs.loc[sample_data.positive_doc_id]['text']

        if max_words:
            hard_negative_doc = truncate_text(hard_negative_doc, max_words)
            positive_doc = truncate_text(positive_doc, max_words)

        positive_sample = IRTrainSample(query=sample_data.query, doc=positive_doc, label=1)
        negative_sample = IRTrainSample(query=sample_data.query, doc=hard_negative_doc, label=0)

    else:
        positive_sample = IRTrainSampleWithoutDoc(query=sample_data.query, doc_id=sample_data.positive_doc_id, label=1)
        negative_sample = IRTrainSampleWithoutDoc(query=sample_data.query, doc_id=hard_negative_id, label=0)

    return positive_sample, negative_sample


def construct_train_set(doc_file, ann_res_file, query_sample_file, out_file, top_n):
    logger.info('Loading data')
    ann_results = read_pickle_file(ann_res_file)
    ann_results = pd.DataFrame(ann_results)
    ann_results = ann_results.set_index('query_id')

    if doc_file is not None:
        docs = pd.read_json(doc_file, lines=True)
        docs = docs.set_index('doc_id')
    else:
        docs = None

    query_samples = read_pickle_file(query_sample_file)

    logger.info('Constructing train samples')

    def get_raw_train_sample(query_sample: QuerySample):
        similar_doc_ids = ann_results.loc[query_sample.query_id].values[0]
        return TrainSampleData(
            similar_doc_ids=similar_doc_ids,
            positive_doc_id=query_sample.positive_doc_id,
            query=query_sample.query,
            query_id=query_sample.query_id
        )

    train_samples_raw = [get_raw_train_sample(sample) for sample in tqdm(query_samples)]
    train_samples = [construct_ir_sample(raw_sample, docs, top_n=top_n)
                     for raw_sample in tqdm(train_samples_raw)]
    train_samples = [sample.__dict__ for sample in itertools.chain.from_iterable(train_samples)]
    write_jsonl_file(train_samples, out_file)


def split_to_df(query_values, doc_values, label_values):
    df = pd.DataFrame(query_values)
    df.columns = ['query']
    df['doc'] = doc_values
    df['label'] = label_values
    return df


def read_df_from_ndjson(file_path, sample_size):
    if sample_size is None:
        return pd.read_json(file_path, lines=True)
    else:
        return pd.read_json(file_path, lines=True, nrows=sample_size)


def read_df_from_csv(file_path, sample_size):
    if sample_size is None:
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(file_path, nrows=sample_size)


def get_train_set_splits(input_file, out_dir, file_type, test_size=0.3, sample_size=None):
    logger.info('Loading data')
    if file_type == 'csv':
        data = read_df_from_csv(input_file, sample_size)
    else:
        data = read_df_from_ndjson(input_file, sample_size)

    logger.info('Splitting data')
    if 'doc' in data.columns:
        data = data[['query', 'doc', 'label']]
        query_train, query_test, doc_train, doc_test, y_train, y_test = train_test_split(
            data['query'].values,
            data['doc'].values,
            data['label'].values,
            test_size=test_size,
            random_state=42
        )
    else:
        query_train, query_test, doc_train, doc_test, y_train, y_test = train_test_split(
            data['query'].values,
            data['doc_id'].values,
            data['label'].values,
            test_size=test_size,
            random_state=42
        )

    train_df = split_to_df(query_train, doc_train, y_train)
    test_df = split_to_df(query_test, doc_test, y_test)

    logger.info('Saving processed data')

    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)



