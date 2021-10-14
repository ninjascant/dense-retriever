import click
from ..estimators.bert_dot import BertDot


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('model_name_or_path', type=str)
@click.argument('dataset_dir', type=str)
@click.argument('out_dir', type=str)
def run_inference_command(model_name_or_path, dataset_dir, out_dir):
    estimator = BertDot(
        model_name_or_path=model_name_or_path,
        train_steps=0,
        batch_size=32,
        accum_steps=0,
        lr=0
    )
    estimator.predict(dataset_dir=dataset_dir, out_dir=out_dir)
