import click
from .ann_index import run_search_prebuilt_index_command, run_search_from_scratch_command, \
    run_evaluation_command
from .data_extraction import construct_query_samples_command
from .inference import run_inference_command

@click.group()
def run():
    pass


run.add_command(run_search_prebuilt_index_command, 'search_from_prebuilt')
run.add_command(run_search_from_scratch_command, 'search_from_scratch')
run.add_command(run_evaluation_command, 'evaluate_index')
run.add_command(construct_query_samples_command, 'construct_query_samples')
run.add_command(run_inference_command, 'run_inference')
