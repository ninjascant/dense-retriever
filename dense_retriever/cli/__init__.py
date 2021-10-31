import click
from .ann_index import run_search_prebuilt_index_command, run_search_from_scratch_command, \
    run_evaluation_command
from .preprocessing import construct_query_samples_command, build_encoding_cache_command, construct_train_set_command, \
    tokenize_train_set_command, tokenize_test_set_command, extract_texts_for_inference_command
from .inference import run_inference_command
from .model_training import train_model_command, train_model_with_refresh_command
from .set_encoding import tokenize_train_triplets_command


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
run.add_command(construct_train_set_command, 'construct_train_set')
run.add_command(tokenize_train_set_command, 'tokenize_train_set')
run.add_command(tokenize_test_set_command, 'tokenize_test_set')
run.add_command(extract_texts_for_inference_command, 'extract_texts_for_inference')
run.add_command(train_model_command, 'train_model')
run.add_command(tokenize_train_triplets_command, 'tokenize_train_triplets')
