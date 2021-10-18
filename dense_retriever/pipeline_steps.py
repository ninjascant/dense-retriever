import datetime
from loguru import logger
import numpy as np
from datasets import load_metric
from .transforms.ann_index import ANNIndex
from .transforms.preprocessing import TrainSetConstructor, TrainSetTokenizer
from .estimators.bert_dot import BertDot
from .utils.file_utils import zip_dir
from .utils.gcs_utils import upload_file_to_gcs


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def compute_metrics(eval_pred):
    metric = load_metric('./metrics/f1_score.py')
    logits, labels = eval_pred
    predictions = softmax(logits)
    return metric.compute(predictions=predictions, references=labels)


def tokenize_train_set(train_set_path, tokenizer_name_or_path, use_cache, out_path):
    transformer = TrainSetTokenizer(tokenizer_name_or_path, max_length=512, padding='max_length', use_cache=use_cache)
    transformer.transform(train_set_path, out_path)


def construct_train_set(search_result_file, query_sample_file, train_docs_file, out_path):
    transformer = TrainSetConstructor(query_sample_file=query_sample_file, train_docs_file=train_docs_file)
    transformer.transform(search_result_file, out_path)


def run_search_from_scratch(
        context_embedding_dir,
        query_embedding_dir,
        out_path,
        embedding_size,
        top_n,
        load_from_sub_dirs,
        index_out_path
):
    transformer = ANNIndex(
        transformer_out_path=index_out_path,
        embedding_size=embedding_size,
        top_n=top_n,
        load_from_sub_dirs=load_from_sub_dirs
    )
    transformer.fit(context_embedding_dir)
    transformer.transform(query_embedding_dir, out_path)


def train_model(model_name, dataset_path, out_dir, batch_size, accum_steps, train_steps=-1, num_epochs=3,
                save_to_gcs=False, log_out_file=None):
    if log_out_file is not None:
        logger.add(log_out_file)
    estimator = BertDot(
        model_name_or_path=model_name,
        train_steps=train_steps,
        num_epochs=num_epochs,
        batch_size=batch_size,
        accum_steps=accum_steps,
    )
    estimator.fit(dataset_dir=dataset_path, model_out_dir=out_dir)

    if save_to_gcs:
        zip_dir(out_dir, f'{out_dir}.tar.gz')
        upload_file_to_gcs('finetuned-models', f'{out_dir}.tar.gz', f'{out_dir}.tar.gz')


def run_inference(model_name_or_path, dataset_dir, out_dir, id_col='doc_id'):
    estimator = BertDot(
        model_name_or_path=model_name_or_path,
        train_steps=0,
        num_epochs=0,
        batch_size=32,
        accum_steps=0,
        lr=0,
        metric_fn=compute_metrics
    )
    estimator.predict(dataset_dir=dataset_dir, out_dir=out_dir, id_col=id_col)


def train_model_with_refresh(
    model_name_or_path,
    init_train_set_dir,
    doc_dataset_dir,
    query_dataset_dir,
    model_out_dir,
    refresh_steps,
    total_steps,
    query_sample_file,
    batch_size,
    accum_steps,
    top_n,
    device,
    num_epochs
):
    refresh_iterations = int(total_steps / refresh_steps) + 1
    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_base_dir = model_out_dir

    for i in range(refresh_iterations):
        model_out_dir = f'{model_base_dir}_{dt}_{i}'
        logger.info(f'Iteration: {i+1}')
        if i == 0:
            train_model(model_name_or_path, init_train_set_dir, model_out_dir, batch_size,
                        accum_steps, log_out_file=f'model-out-{i}.log', save_to_gcs=True, num_epochs=num_epochs)
        else:
            model_out_prev_epoch = f'{model_base_dir}_{dt}_{i - 1}'
            logger.info('Updating doc embeddings')
            run_inference(model_out_prev_epoch, doc_dataset_dir, 'model_outputs/doc_embeddings', 'doc_id')
            logger.info('Updating query embeddings')
            run_inference(model_out_prev_epoch, query_dataset_dir, 'model_outputs/query_embeddings', 'id')

            logger.info('Updating ANN index')
            run_search_from_scratch('model_outputs/doc_embeddings', 'model_outputs/query_embeddings',
                                    'model_outputs/query_similar_docs.pkl', embedding_size=312, top_n=top_n,
                                    index_out_path=None, load_from_sub_dirs=False)

            logger.info('Updating train set')
            construct_train_set('model_outputs/query_similar_docs.pkl', query_sample_file, None,
                                'model_outputs/train_set.json')

            tokenize_train_set('model_outputs/train_set.json', model_name_or_path, use_cache=True,
                               out_path='model_outputs/dataset_refreshed')

            logger.info('Continue model training')
            train_model(model_out_prev_epoch, 'model_outputs/dataset_refreshed', model_out_dir,
                        batch_size, accum_steps, log_out_file=f'model-out-{i}.log', save_to_gcs=True,
                        train_steps=refresh_steps)
