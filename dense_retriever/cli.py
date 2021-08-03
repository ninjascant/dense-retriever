import click
from .tokenize_dataset import tokenize_dataset
from .data_extraction import extract
from .inference import run_inference
from .db_ops import db_ops
from .ann_index import build_index, validate


@click.group()
def run():
    pass


run.add_command(tokenize_dataset, 'tokenize')
run.add_command(extract, 'extract')
run.add_command(run_inference, 'inference')
run.add_command(db_ops, 'db_ops')
run.add_command(build_index, 'build_index')
run.add_command(validate, 'validate')
