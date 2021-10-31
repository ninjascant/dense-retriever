import click
from ..pipeline_steps.set_encoding import tokenize_train_triplets


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('tokenizer_name_or_path', type=str)
@click.argument('triplet_file_path', type=str)
@click.argument('out_path', type=str)
def tokenize_train_triplets_command(tokenizer_name_or_path, triplet_file_path, out_path):
    tokenize_train_triplets(
        tokenizer_name_or_path=tokenizer_name_or_path,
        triplet_file_path=triplet_file_path,
        out_path=out_path)
