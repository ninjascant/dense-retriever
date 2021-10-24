import click
from ..pipeline_steps.training import train_model, train_model_with_refresh


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('model_name_or_path', type=str)
@click.argument('model_type', type=str)
@click.argument('train_set_path', type=str)
@click.argument('model_out_dir', type=str)
@click.option('-t', '--total-steps', type=int, default=-1)
@click.option('-e', '--num-epochs', type=int, default=1)
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('-i', '--in-batch-neg', type=bool, default=False)
def train_model_command(model_name_or_path, model_type, train_set_path, model_out_dir, total_steps, num_epochs,
                        batch_size, in_batch_neg):
    train_model(
        model_name=model_name_or_path,
        model_type=model_type,
        dataset_path=train_set_path,
        out_dir=model_out_dir,
        train_steps=total_steps,
        num_epochs=num_epochs,
        accum_steps=4,
        batch_size=batch_size,
        in_batch_neg=in_batch_neg
    )


@click.command()
@click.argument('model_name_or_path', type=str)
@click.argument('init_train_set_dir', type=str)
@click.argument('doc_dataset_dir', type=str)
@click.argument('query_dataset_dir', type=str)
@click.argument('query_sample_file', type=str)
@click.argument('model_out_dir', type=str)
@click.argument('refresh_steps', type=int)
@click.argument('total_steps', type=int)
@click.argument('num_epochs', type=int)
@click.option('-t', '--top-n', type=int, default=50)
def train_model_with_refresh_command(model_name_or_path, init_train_set_dir, doc_dataset_dir, query_dataset_dir,
                                     query_sample_file, model_out_dir, refresh_steps, total_steps, num_epochs, top_n):
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
        num_epochs=num_epochs
    )
