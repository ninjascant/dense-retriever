import click
from ..pipeline_steps import run_inference


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('model_name_or_path', type=str)
@click.argument('dataset_dir', type=str)
@click.argument('out_dir', type=str)
def run_inference_command(model_name_or_path, dataset_dir, out_dir):
    run_inference(model_name_or_path, dataset_dir, out_dir)
