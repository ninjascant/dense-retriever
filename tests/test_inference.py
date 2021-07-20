from pathlib import Path
import numpy as np
import torch
from dense_retriever.inference import prepare_dataset, prepare_dataloader, post_process_dataset, InferenceRunner, \
    _run_inference


def test_prepare_dataset():
    prepared_dataset = prepare_dataset('tests/files/test_tokenized_dataset', ['input_ids', 'attention_mask'])

    assert sorted(prepared_dataset['train'].column_names) == ['attention_mask', 'doc_id', 'input_ids']
    assert torch.is_tensor(prepared_dataset['train'][0]['input_ids'])
    assert torch.is_tensor(prepared_dataset['train'][0]['attention_mask'])
    assert list(prepared_dataset.keys()) == ['train']


def test_post_process_dataset():
    test_runner = InferenceRunner('rdot_nll', '../ance_model', 'cpu')
    dataset = prepare_dataset('tests/files/test_tokenized_dataset', ['input_ids', 'attention_mask'])
    dataloader = prepare_dataloader(dataset, 32)
    embeddings = test_runner.transform(dataloader)
    post_processed_dataset = post_process_dataset(dataset, embeddings, ['input_ids', 'attention_mask'])

    assert sorted(post_processed_dataset['train'].column_names) == ['doc_id', 'embedding']
    assert type(post_processed_dataset['train'][0]['embedding']) == np.ndarray


def test_run_inference_without_zipping():
    test_embed_dir = Path('tests/files/test_embedding_dataset')
    test_embed_dir.unlink(missing_ok=True)

    _run_inference(
        'tests/files/test_tokenized_dataset',
        'rdot_nll',
        '../ance_model',
        'tests/files/test_embedding_dataset',
        'input_ids,attention_mask',
        32,
        False,
        None,
        None,
        'cpu'
    )

    assert test_embed_dir.is_dir()

