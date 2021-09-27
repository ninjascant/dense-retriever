import click
from .preprocessing import tokenize_dataset
from .data_extraction import extract
from .inference import run_inference
from .ann_index import build_index, validate, get_train_samples
from .train import train_model


@click.command()
@click.argument('train_file_path', type=str)
@click.argument('out_dir', type=str)
@click.argument('model_name', type=str)
@click.option('-t', '--test-file-path', type=str, default=None)
@click.option('-z', '--zip-path', type=str, default=None)
@click.option('-m', '--max-length', type=int, default=512)
@click.option('-p', '--padding', type=str, default='max_length')
def tokenize_ir_dataset(train_file_path, out_dir, model_name, test_file_path, zip_path, max_length, padding):
    tokenize_dataset(train_file_path, test_file_path, out_dir, model_name, zip_path, max_length, padding)


@click.group()
def run():
    pass


run.add_command(extract, 'extract')
run.add_command(run_inference, 'inference')
run.add_command(build_index, 'build_index')
run.add_command(validate, 'validate')
run.add_command(get_train_samples, 'train_samples')
run.add_command(train_model, 'train_model')
