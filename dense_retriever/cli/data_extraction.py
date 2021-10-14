import click
from ..transforms.data_extraction import QuerySampleConstructor


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('qrel_path', type=str)
@click.argument('query_path', type=str)
@click.argument('out_path', type=str)
def construct_query_samples_command(qrel_path, query_path, out_path):
    transformer = QuerySampleConstructor(qrel_path=qrel_path)
    transformer.transform(query_path, out_path)
