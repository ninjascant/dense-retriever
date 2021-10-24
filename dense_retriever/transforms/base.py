from typing import Any, Union
from loguru import logger


class BaseTransform:
    def __init__(self, transformer_out_path: Union[str, None]):
        self._transformer_out_path = transformer_out_path
        self.transformer = None

    def _load_input_data(self, input_path: str):
        """
        A base method that handles loading input data from disk

        :param input_path: path to input data
        :return: input data
        """
        raise NotImplementedError('You need to implement this method')

    def _save_transformed_data(self, transformed_data: Any, out_path: str):
        raise NotImplementedError('You need to implement this method')

    def _save_transformer(self, out_path: str):
        raise NotImplementedError('You need to implement this method')

    def _load_transformer(self, input_path: str):
        raise NotImplementedError('You need to implement this method')

    def _fit_transformer_fn(self, input_data: Any):
        raise NotImplementedError('You need to implement this method')

    def _transform_fn(self, input_data: Any):
        raise NotImplementedError('You need to implement this method')

    def fit(self, input_path: str):
        logger.info('Loading input data')
        input_data = self._load_input_data(input_path)
        logger.info('Fitting transformer')
        self.transformer = self._fit_transformer_fn(input_data)
        if self._transformer_out_path is not None:
            logger.info('Saving transformer')
            self._save_transformer(self._transformer_out_path)
        return self

    def transform(self, input_path: str, out_path: str, transformer_load_path: Union[str, None] = None):
        if transformer_load_path is not None:
            logger.info('Loading transformer')
            self._load_transformer(transformer_load_path)
        logger.info('Loading input data')
        input_data = self._load_input_data(input_path)
        logger.info('Transforming data')
        transformed_data = self._transform_fn(input_data)
        logger.info('Saving transformed data')
        self._save_transformed_data(transformed_data, out_path)
        return transformed_data
