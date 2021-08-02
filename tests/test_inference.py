import os
import torch
from dense_retriever.inference import prepare_dataset, _run_inference


def test_prepare_dataset():
    prepared_dataset = prepare_dataset('tests/files/test_tokenized_dataset', ['input_ids', 'attention_mask'])

    assert sorted(prepared_dataset['train'].column_names) == ['attention_mask', 'doc_id', 'input_ids']
    assert torch.is_tensor(prepared_dataset['train'][0]['input_ids'])
    assert torch.is_tensor(prepared_dataset['train'][0]['attention_mask'])
    assert list(prepared_dataset.keys()) == ['train']


def test_run_inference_without_zipping():
    test_embed_dir = 'tests/files/test_embedding_dataset'

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
        'cpu',
        overwrite=True
    )

    assert os.path.isdir(test_embed_dir)
    assert sorted(os.listdir(test_embed_dir)) == ['embeddings.npy', 'ids.json']
