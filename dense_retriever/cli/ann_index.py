import click
from ..transforms.ann_index import ANNIndex, SearchEvaluator
from ..pipeline_steps import run_search_from_scratch


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('context_embedding_dir', type=str)
@click.argument('query_embedding_dir', type=str)
@click.argument('out_path', type=str)
@click.argument('embedding_size', type=int)
@click.argument('top_n', type=int)
@click.option('-s', '--load-from-sub-dirs', type=bool, default=False)
@click.option('-i', '--index-out-path', type=str, default=None)
def run_search_from_scratch_command(
        context_embedding_dir,
        query_embedding_dir,
        out_path,
        embedding_size,
        top_n,
        load_from_sub_dirs,
        index_out_path
):
    run_search_from_scratch(context_embedding_dir, query_embedding_dir, out_path, embedding_size, top_n,
                            load_from_sub_dirs, index_out_path)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('index_path', type=str)
@click.argument('query_embedding_dir', type=str)
@click.argument('out_path', type=str)
@click.argument('embedding_size', type=int)
@click.argument('top_n', type=int)
@click.option('-s', '--load-from-sub-dirs', type=bool, default=False)
def run_search_prebuilt_index_command(
        index_path,
        query_embedding_dir,
        out_path,
        embedding_size,
        top_n,
        load_from_sub_dirs,
):
    transformer = ANNIndex(
        transformer_out_path=None,
        embedding_size=embedding_size,
        top_n=top_n,
        load_from_sub_dirs=load_from_sub_dirs,
    )
    transformer.transform(query_embedding_dir, out_path, transformer_load_path=index_path)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('query_sample_file', type=str)
@click.argument('search_result_file', type=str)
@click.argument('out_file', type=str)
def run_evaluation_command(
        query_sample_file,
        search_result_file,
        out_file
):
    transformer = SearchEvaluator(transformer_out_path=None, query_sample_file=query_sample_file)
    transformer.transform(input_path=search_result_file, out_path=out_file)