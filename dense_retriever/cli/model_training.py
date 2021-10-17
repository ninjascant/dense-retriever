import click
from ..pipeline_steps import train_model_with_refresh


@click.command()
@click.argument('model_name_or_path', type=str)
@click.argument('init_train_set_dir', type=str)
@click.argument('doc_dataset_dir', type=str)
@click.argument('query_dataset_dir', type=str)
@click.argument('query_sample_file', type=str)
@click.argument('model_out_dir', type=str)
@click.argument('refresh_steps', type=int)
@click.argument('total_steps', type=int)
@click.argument('warmup_steps', type=int)
@click.option('-t', '--top-n', type=int, default=50)
def train_model_with_refresh_command(model_name_or_path, init_train_set_dir, doc_dataset_dir, query_dataset_dir,
                                     query_sample_file, model_out_dir, refresh_steps, total_steps, warmup_steps, top_n):
    train_model_with_refresh(
        model_name_or_path=model_name_or_path,
        init_train_set_dir=init_train_set_dir,
        doc_dataset_dir=doc_dataset_dir,
        query_dataset_dir=query_dataset_dir,
        model_out_dir=model_out_dir,
        refresh_steps=refresh_steps,
        total_steps=total_steps,
        query_sample_file=query_sample_file,
        batch_size=8,
        accum_steps=3,
        top_n=top_n,
        device='cuda',
        warmup_steps=warmup_steps
    )
