import itertools
from typing import Any, List
from tqdm.auto import tqdm
from loguru import logger
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast
from datasets import load_dataset, Dataset
from .base import BaseTransform
from ..data_model import QuerySample, TrainSampleData, IRTrainSample, IRTrainSampleWithoutDoc
from ..utils.redis_utils import RedisClient
from ..utils.file_utils import read_pickle_file, write_pickle_file, write_jsonl_file


def _rename_torch_columns(dataset, column_name):
    dataset = dataset.rename_column('input_ids', f'{column_name}_input_ids')
    dataset = dataset.rename_column('attention_mask', f'{column_name}_attention_mask')
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


def truncate_text(text, max_words):
    return ' '.join(text.split()[:max_words])


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


class QuerySampleConstructor(BaseTransform):
    def __init__(self, qrel_path: str, transformer_out_path: None = None):
        super(QuerySampleConstructor, self).__init__(transformer_out_path)

        self._qrel_path = qrel_path

    def _load_input_data(self, input_path: str):
        qrels = pd.read_csv(self._qrel_path, sep=' ')
        queries = pd.read_csv(input_path, sep='\t')
        return queries, qrels

    def _transform_fn(self, input_data: Any):
        queries, qrels = input_data

        queries.columns = ['qid', 'text']
        qrels.columns = ['qid', 'none', 'doc_id', 'none1']

        queries = queries.merge(qrels, on='qid', how='inner')

        query_samples = [QuerySample(query=row['text'], query_id=row['qid'], positive_doc_id=row['doc_id'])
                         for i, row in queries.iterrows()]
        return query_samples

    def _save_transformed_data(self, transformed_data: List[QuerySample], out_path: str):
        write_pickle_file(out_path, transformed_data)

    def _save_transformer(self, out_path: str):
        pass

    def _load_transformer(self, input_path: str):
        pass

    def _fit_transformer_fn(self, input_data: Any):
        pass


class TrainSetConstructor(BaseTransform):
    def __init__(self, query_sample_file, train_docs_file=None):
        super(TrainSetConstructor, self).__init__(transformer_out_path=None)

        self._query_sample_file = query_sample_file
        self._train_docs_file = train_docs_file

    def _load_input_data(self, input_path: str):
        search_results = read_pickle_file(input_path)
        query_samples = read_pickle_file(self._query_sample_file)
        search_results = pd.DataFrame(search_results)
        query_samples = pd.DataFrame(query_samples)
        search_samples = query_samples.merge(search_results, on='query_id', how='inner')
        return search_samples

    def _transform_fn(self, input_data: pd.DataFrame):
        if self._train_docs_file is not None:
            docs = pd.read_json(self._train_docs_file, lines=True)
            docs = docs.set_index('doc_id')
        else:
            docs = None

        train_samples_raw = [
            TrainSampleData(
                similar_doc_ids=row['search_results'],
                positive_doc_id=row['positive_doc_id'],
                query=row['query'], query_id=row['query_id']
            ) for i, row in input_data.iterrows()]

        top_n = len(train_samples_raw[0].similar_doc_ids)
        train_samples = [construct_ir_sample(raw_sample, docs, top_n=top_n)
                         for raw_sample in tqdm(train_samples_raw)]
        train_samples = [sample.__dict__ for sample in itertools.chain.from_iterable(train_samples)]
        return train_samples

    def _save_transformed_data(self, transformed_data: List[dict], out_path: str):
        write_jsonl_file(transformed_data, out_path)

    def _save_transformer(self, out_path: str):
        pass

    def _load_transformer(self, input_path: str):
        pass

    def _fit_transformer_fn(self, input_data: Any):
        pass


class TrainSetTokenizer(BaseTransform):
    def __init__(self, tokenizer_name_or_path: str, max_length: int, padding: str, use_cache: bool):
        super(TrainSetTokenizer, self).__init__(transformer_out_path=None)

        self._use_cache = use_cache
        self.max_length = max_length
        self.padding = padding

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name_or_path)

    def _load_input_data(self, input_path: str):
        dataset = load_dataset('json', data_files=[input_path])
        dataset = dataset.train_test_split(test_size=0.2)
        return dataset

    def _transform_fn(self, input_data: Any):
        if not self._use_cache:
            logger.info('Tokenizing dataset')
            encoded_dataset = _encode_text_column(input_data, self.tokenizer, 'query', self.max_length, self.padding)
            encoded_dataset = _encode_text_column(encoded_dataset, self.tokenizer, 'doc', self.max_length, self.padding)
        else:
            encoded_dataset = _set_encoding_from_cache(input_data, 'query', True)
            encoded_dataset = _set_encoding_from_cache(encoded_dataset, 'doc', True)

        return encoded_dataset

    def _save_transformed_data(self, transformed_data: Dataset, out_path: str):
        transformed_data.save_to_disk(out_path)

    def _save_transformer(self, out_path: str):
        pass

    def _load_transformer(self, input_path: str):
        pass

    def _fit_transformer_fn(self, input_data: Any):
        pass
