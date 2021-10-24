import datetime
from loguru import logger
import numpy as np
from datasets import load_metric
from ..estimators.bert_dot import BertDot
from .inference import run_inference
from .ann_search import run_search_from_scratch
from .preprocessing import tokenize_train_set, construct_train_set
from ..utils.file_utils import zip_dir
from ..utils.gcs_utils import upload_file_to_gcs


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def compute_metrics(eval_pred):
    metric = load_metric('dense-retriever/dense_retriever/metrics/f1_score.py')
    logits, labels = eval_pred
    predictions = softmax(logits)
    return metric.compute(predictions=predictions, references=labels)


def train_model(model_name, model_type, dataset_path, out_dir, batch_size, accum_steps, train_steps=-1, num_epochs=3,
                save_to_gcs=False, log_out_file=None, continue_train=False, save_steps=None):
    if log_out_file is not None:
        logger.add(log_out_file)
    estimator = BertDot(
        model_name_or_path=model_name,
        model_type=model_type,
        train_steps=train_steps,
        num_epochs=num_epochs,
        batch_size=batch_size,
        accum_steps=accum_steps,
        continue_train=continue_train,
        metric_fn=compute_metrics,
        save_steps=save_steps,
        in_batch_neg=True
    )
    estimator.fit(dataset_dir=dataset_path, model_out_dir=out_dir)

    if save_to_gcs:
        zip_dir(out_dir, f'{out_dir}.tar.gz')
        upload_file_to_gcs('finetuned-models', f'{out_dir}.tar.gz', f'{out_dir}.tar.gz')


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
                        train_steps=refresh_steps, save_steps=refresh_steps, continue_train=True)

