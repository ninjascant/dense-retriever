from typing import Any, List
from loguru import logger
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast
from datasets import load_dataset, Dataset
from .base import BaseTransform
from ..data_model import QuerySample, IRTrainSample, IRTrainSampleWithoutDoc
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
        doc_id = example['doc_id']
        encoding = client.read(doc_id)
        return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask']}

    encoded_dataset = dataset.map(get_encoding)
    encoded_dataset = encoded_dataset.remove_columns('doc_id')
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

    @staticmethod
    def _get_ann_hard_negative_for_query(row):
        ann_search_res = [ctx_id for ctx_id in row['search_results'] if ctx_id != row['positive_doc_id']]
        hard_negative_id = np.random.choice(ann_search_res)
        row['hard_negative_id'] = hard_negative_id
        return row

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
        else:
            docs = None
        logger.info('Loaded docs')

        queries_with_neg = input_data.apply(self._get_ann_hard_negative_for_query, axis=1)

        if docs is not None:
            queries_with_ctx = queries_with_neg.merge(docs[['doc_id', 'text']], left_on='hard_negative_id',
                                                      right_on='doc_id',
                                                      how='inner')
            queries_with_ctx = queries_with_ctx.rename(columns={'text': 'neg_ctx'})
            queries_with_ctx = queries_with_ctx.drop(['doc_id'], axis=1)

            queries_with_ctx = queries_with_ctx.merge(docs[['doc_id', 'text']], left_on='positive_doc_id',
                                                      right_on='doc_id',
                                                      how='inner').sample(frac=1).reset_index(drop=True)
            queries_with_ctx = queries_with_ctx.rename(columns={'text': 'pos_ctx'})
            queries_with_ctx = queries_with_ctx.drop(
                ['doc_id', 'hard_negative_id', 'positive_doc_id', 'query_id', 'search_results'], axis=1)
            pos_sample_df = queries_with_ctx[['query', 'pos_ctx']]
            pos_sample_df['label'] = 1
            pos_sample_df = pos_sample_df.rename(columns={'pos_ctx': 'context'})
            pos_samples = pos_sample_df.to_dict(orient='records')

            neg_sample_df = queries_with_ctx[['query', 'neg_ctx']]
            neg_sample_df = neg_sample_df.rename(columns={'neg_ctx': 'context'})
            neg_sample_df['label'] = 0
            neg_samples = neg_sample_df.to_dict(orient='records')

            train_samples = pos_samples + neg_samples
        else:
            pos_sample_df = queries_with_neg[['query', 'positive_doc_id']]
            pos_sample_df['label'] = 1
            pos_sample_df = pos_sample_df.rename(columns={'positive_doc_id': 'doc_id'})
            pos_samples = pos_sample_df.to_dict(orient='records')

            neg_sample_df = queries_with_neg[['query', 'hard_negative_id']]
            neg_sample_df['label'] = 0
            neg_sample_df = neg_sample_df.rename(columns={'hard_negative_id': 'doc_id'})
            neg_samples = neg_sample_df.to_dict(orient='records')
            train_samples = pos_samples + neg_samples
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
        dataset = load_dataset('json', data_files=[input_path])['train']
        dataset = dataset.train_test_split(test_size=0.2)
        return dataset

    def _transform_fn(self, input_data: Any):
        if not self._use_cache:
            logger.info('Tokenizing dataset')
            encoded_dataset = _encode_text_column(input_data, self.tokenizer, 'query', self.max_length, self.padding)
            encoded_dataset = _encode_text_column(encoded_dataset, self.tokenizer, 'context', self.max_length,
                                                  self.padding)
        else:
            encoded_dataset = _encode_text_column(input_data, self.tokenizer, 'query', 100, 'max_length')
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


class TestSetTokenizer(BaseTransform):
    def __init__(self, tokenizer_name_or_path: str, max_length: int = 512, padding: str = 'max_length',
                 text_column: str = 'context'):
        super(TestSetTokenizer, self).__init__(transformer_out_path=None)

        self.max_length = max_length
        self.padding = padding

        self._text_column = text_column

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name_or_path)

    def _load_input_data(self, input_path: str):
        dataset = load_dataset('json', data_files={'test': input_path})
        return dataset

    def _save_transformed_data(self, transformed_data: Dataset, out_path: str):
        transformed_data.save_to_disk(out_path)

    def _transform_fn(self, input_data: Any):
        encoded_dataset = _encode_text_column(input_data, self.tokenizer, self._text_column, self.max_length,
                                              self.padding, rename_cols=False)
        return encoded_dataset

    def _fit_transformer_fn(self, input_data: Any):
        pass

    def _load_transformer(self, input_path: str):
        pass

    def _save_transformer(self, out_path: str):
        pass
