import click
from ..transforms.preprocessing import QuerySampleConstructor
from ..transforms.encoding_cache import EncodingCacheBuilder
from ..pipeline_steps import construct_train_set, tokenize_train_set, tokenize_test_set


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('qrel_path', type=str)
@click.argument('query_path', type=str)
@click.argument('out_path', type=str)
def construct_query_samples_command(qrel_path, query_path, out_path):
    transformer = QuerySampleConstructor(qrel_path=qrel_path)
    transformer.transform(query_path, out_path)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('encoding_dataset_dir', type=str)
def build_encoding_cache_command(encoding_dataset_dir):
    transformer = EncodingCacheBuilder()
    transformer.transform(encoding_dataset_dir, '')


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('search_result_file', type=str)
@click.argument('query_sample_file', type=str)
@click.argument('out_path', type=str)
@click.option('-t', '--train-docs-file', type=str, default=None)
def construct_train_set_command(search_result_file, query_sample_file, out_path, train_docs_file):
    construct_train_set(
        search_result_file=search_result_file,
        query_sample_file=query_sample_file,
        train_docs_file=train_docs_file,
        out_path=out_path)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('train_set_path', type=str)
@click.argument('tokenizer_name_or_path', type=str)
@click.argument('out_path', type=str)
@click.option('-c', '--use_cache', type=bool, default=False)
def tokenize_train_set_command(train_set_path, tokenizer_name_or_path, out_path, use_cache):
    tokenize_train_set(
        train_set_path=train_set_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        use_cache=use_cache,
        out_path=out_path
    )


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('test_set_path', type=str)
@click.argument('tokenizer_name_or_path', type=str)
@click.argument('out_path', type=str)
@click.option('-t', '--text-col', type=str, default='context')
def tokenize_test_set_command(test_set_path, tokenizer_name_or_path, out_path, text_col):
    tokenize_test_set(
        test_set_path=test_set_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        out_path=out_path,
        text_col_name=text_col
    )
