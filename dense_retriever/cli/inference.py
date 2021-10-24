import click
from ..pipeline_steps.inference import run_inference


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('model_name_or_path', type=str)
@click.argument('model_type', type=str)
@click.argument('dataset_dir', type=str)
@click.argument('out_dir', type=str)
@click.option('-i', '--id-col', type=str, default='doc_id')
def run_inference_command(model_name_or_path, model_type, dataset_dir, out_dir, id_col):
    run_inference(model_name_or_path, model_type, dataset_dir, out_dir, id_col)
