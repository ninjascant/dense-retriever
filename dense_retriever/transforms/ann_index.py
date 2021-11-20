import os
from typing import Union, Any
from tqdm.auto import tqdm
from loguru import logger
import numpy as np
import pandas as pd
import faiss
from .base import BaseTransform
from ..data_model import ANNSearchRes
from ..utils.file_utils import load_json_file, read_pickle_file, write_pickle_file


def _convert_ids_to_int(ids):
    is_str = type(ids[0]) is str
    if is_str:
        ids = [int(doc_id[1:]) for doc_id in ids]
    return ids


def convert_idx_to_id(indices):
    return [f'D{idx}' for idx in indices]


class ANNIndex(BaseTransform):
    def __init__(
            self, 
            transformer_out_path: Union[str, None], 
            embedding_size: int, top_n: int, 
            load_from_sub_dirs: bool = False
    ):
        """
        Transformer that performs standard ANN index operations: load context and query embeddings, 
        build index and run search over it
         
        :param transformer_out_path: path to file to which index will be saved. If None, index will not be saved
        :param embedding_size: number of embedding dimensions
        :param top_n: number of result to return while searching
        :param load_from_sub_dirs: whether embedding files are located in sub dirs
        """
        super().__init__(transformer_out_path)

        self.embedding_size = embedding_size
        self.top_n = top_n
        self.load_from_sub_dirs = load_from_sub_dirs

    @staticmethod
    def _load_embeddings_from_sub_dirs(input_dir):
        sub_dirs = [item for item in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, item))]
        num_embed_sub_dirs = len(sub_dirs)

        total_embeddings = []
        total_ids = []

        for i in tqdm(range(num_embed_sub_dirs)):
            embeddings = np.load(os.path.join(input_dir, f'embeddings_{i}', 'embeddings.npy')).astype(np.float32)

            ids = load_json_file(os.path.join(input_dir, f'embeddings_{i}', 'ids.json'))
            ids = [int(doc_id[1:]) for doc_id in ids]
            ids = np.array(ids).astype(np.int64)

            total_embeddings.append(embeddings)
            total_ids.append(ids)

        total_embeddings = np.vstack(total_embeddings)
        total_ids = np.vstack(total_ids)

        return total_embeddings, total_ids

    @staticmethod
    def _load_embeddings_from_single_file(input_dir):
        ids = load_json_file(os.path.join(input_dir, 'ids.json'))
        ids = _convert_ids_to_int(ids)
        ids = np.array(ids).astype(np.int64)

        embeddings = np.load(os.path.join(input_dir, 'embeddings.npy')).astype(np.float32)

        return embeddings, ids

    def _load_input_data(self, input_path):
        if self.load_from_sub_dirs:
            embeddings, ids = self._load_embeddings_from_sub_dirs(input_path)
        else:
            embeddings, ids = self._load_embeddings_from_single_file(input_path)

        return embeddings, ids

    def _fit_transformer_fn(self, input_data):
        index = faiss.index_factory(self.embedding_size, 'IDMap,Flat', faiss.METRIC_L2)
        embeddings, ids = input_data
        index.add_with_ids(embeddings, ids)
        return index

    def _transform_fn(self, input_data):
        embeddings, ids = input_data
        _, res_idx = self.transformer.search(embeddings, self.top_n)
        res_ids = [convert_idx_to_id(indices) for indices in res_idx]
        search_results = [ANNSearchRes(ids[i], result) for i, result in enumerate(res_ids)]
        return search_results

    def _save_transformed_data(self, transformed_data, out_path):
        write_pickle_file(out_path, transformed_data)

    def _save_transformer(self, out_path):
        faiss.write_index(self.transformer, out_path)

    def _load_transformer(self, input_path):
        self.transformer = faiss.read_index(input_path)


class SearchEvaluator(BaseTransform):
    def __init__(self, transformer_out_path: None, query_sample_file: str):
        """
        Transformer that evaluates embedding performance by calculating MRR over ANN index search results

        :param transformer_out_path: leave it empty since SearchEvaluator doesn't have a transformer
        :param query_sample_file: path to pickle file with query samples (see data_model.QuerySample class for more
        details
        """
        super(SearchEvaluator, self).__init__(transformer_out_path)
        self._query_sample_file = query_sample_file

    @staticmethod
    def _calc_reciprocal_rank(row):
        if row['positive_doc_id'] in row['search_results']:
            reciprocal_rank = 1 / (row['search_results'].index(row['positive_doc_id']) + 1)
        else:
            reciprocal_rank = 0
        row['reciprocal_rank'] = reciprocal_rank
        return row

    def _load_input_data(self, input_path: str):
        search_results = read_pickle_file(input_path)
        query_samples = read_pickle_file(self._query_sample_file)
        search_results = pd.DataFrame(search_results)
        query_samples = pd.DataFrame(query_samples)
        search_samples = query_samples.merge(search_results, on='query_id', how='inner')
        return search_samples

    def _save_transformed_data(self, transformed_data: pd.DataFrame, out_path: str):
        transformed_data.to_csv(out_path, index=False)

    def _save_transformer(self, out_path: str):
        pass

    def _load_transformer(self, input_path: str):
        pass

    def _fit_transformer_fn(self, input_data: Any):
        return None

    def _transform_fn(self, input_data):
        search_results_with_ranks = input_data.apply(self._calc_reciprocal_rank, axis=1)
        top_n = len(search_results_with_ranks.iloc[0]['search_results'])
        mrr = search_results_with_ranks['reciprocal_rank'].mean()
        mrr = '%.6f' % mrr
        logger.info(f'MRR@{top_n}: {mrr}')
        return search_results_with_ranks
