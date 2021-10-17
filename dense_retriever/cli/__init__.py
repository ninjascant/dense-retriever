import click
from .ann_index import run_search_prebuilt_index_command, run_search_from_scratch_command, \
    run_evaluation_command
from .preprocessing import construct_query_samples_command, build_encoding_cache_command
from .inference import run_inference_command
from .model_training import train_model_with_refresh_command


@click.group()
def run():
    pass


run.add_command(run_search_prebuilt_index_command, 'search_from_prebuilt')
run.add_command(run_search_from_scratch_command, 'search_from_scratch')
run.add_command(run_evaluation_command, 'evaluate_index')
run.add_command(construct_query_samples_command, 'construct_query_samples')
run.add_command(run_inference_command, 'run_inference')
run.add_command(train_model_with_refresh_command, 'train_model_with_refresh')
run.add_command(build_encoding_cache_command, 'build_encoding_cache')
