from typing import Any, List
import pandas as pd
from .base import BaseTransform
from ..data_model import QuerySample
from ..utils.file_utils import write_pickle_file


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
