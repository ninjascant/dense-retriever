import click
from .preprocessing import tokenize_train_dataset, tokenize_test_dataset, get_train_set_splits, \
    construct_train_set, get_similar_docs, truncate_docs, prepare_encoding_cache
from .preprocessing.data_tokenization import export_encoding_to_redis
from .data_extraction import extract
from .data_extraction.msmarco import join_query_qrels
from .inference import run_inference
from .ann_index import build_index, build_index_single_file, validate, get_train_samples
from .train import train_model, train_model_with_refresh


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('index_file', type=str)
@click.argument('query_embed_dir', type=str)
@click.argument('out_file', type=str)
@click.option('-t', '--top-n', type=int, default=100)
def get_query_similar_docs(index_file, query_embed_dir, out_file, top_n):
    get_similar_docs(
        index_file=index_file,
        query_embed_dir=query_embed_dir,
        out_file=out_file,
        top_n=top_n
    )


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('doc_file', type=str)
@click.argument('ann_res_file', type=str)
@click.argument('query_file', type=str)
@click.argument('out_file', type=str)
@click.option('-t', '--top-n', type=int, default=100)
def construct_train_dataset(doc_file, ann_res_file, query_file, out_file, top_n):
    construct_train_set(
        doc_file=doc_file,
        ann_res_file=ann_res_file,
        query_sample_file=query_file,
        out_file=out_file,
        top_n=top_n
    )


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('input_file', type=str)
@click.argument('out_dir', type=str)
@click.option('-f', '--file-type', type=str, default='csv')
@click.option('-t', '--test-size', type=float, default=0.3)
@click.option('-s', '--sample-size', type=float, default=None)
def get_train_splits(input_file, out_dir, file_type, test_size, sample_size):
    get_train_set_splits(
        input_file=input_file,
        out_dir=out_dir,
        file_type=file_type,
        test_size=test_size,
        sample_size=sample_size
    )


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('train_file_path', type=str)
@click.argument('out_dir', type=str)
@click.argument('model_name', type=str)
@click.option('-f', '--file-type', type=str, default='csv')
@click.option('-t', '--test-file-path', type=str, default=None)
@click.option('-z', '--zip-path', type=str, default=None)
@click.option('-m', '--max-length', type=int, default=512)
@click.option('-p', '--padding', type=str, default='max_length')
def tokenize_train_data(train_file_path, out_dir, model_name, file_type, test_file_path, zip_path, max_length, padding):
    tokenize_train_dataset(
        train_file_path,
        test_file_path,
        out_dir,
        model_name,
        file_type,
        zip_path,
        max_length,
        padding
    )


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('input_file', type=str)
@click.argument('out_dir', type=str)
@click.argument('model_name', type=str)
@click.option('-f', '--file-type', type=str, default='csv')
@click.option('-z', '--zip-path', type=str, default=None)
@click.option('-m', '--max-length', type=int, default=512)
@click.option('-p', '--padding', type=str, default='max_length')
def tokenize_test_data(input_file, out_dir, model_name, file_type, zip_path, max_length, padding):
    tokenize_test_dataset(
        input_file,
        out_dir,
        model_name,
        file_type,
        zip_path,
        max_length,
        padding
    )


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('input_file', type=str)
@click.argument('out_file', type=str)
def truncate(input_file, out_file):
    truncate_docs(input_file, out_file)


@click.command()
@click.argument('input_file')
@click.argument('out_file')
def prepare_encoding_cache_command(input_file, out_file):
    prepare_encoding_cache(input_file, out_file)


@click.command()
@click.argument('encoding_dataset_dir')
def export_encodings_command(encoding_dataset_dir):
    export_encoding_to_redis(encoding_dataset_dir)


@click.group()
def run():
    pass


run.add_command(extract, 'extract')

run.add_command(get_query_similar_docs, 'get_query_similar_docs')
run.add_command(construct_train_dataset, 'construct_train_dataset')
run.add_command(get_train_splits, 'get_train_splits')
run.add_command(tokenize_train_data, 'tokenize_train_data')
run.add_command(tokenize_test_data, 'tokenize_test_data')
run.add_command(truncate, 'truncate')

run.add_command(run_inference, 'inference')
run.add_command(build_index, 'build_index')
run.add_command(build_index_single_file, 'build_index_single_file')
run.add_command(validate, 'validate')
run.add_command(get_train_samples, 'train_samples')
run.add_command(train_model, 'train_model')
run.add_command(train_model_with_refresh, 'train_model_with_refresh')
run.add_command(prepare_encoding_cache_command, 'prepare_encoding_cache')
run.add_command(join_query_qrels, 'join_query_qrels')
run.add_command(export_encodings_command, 'export_encodings')
