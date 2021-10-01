import click
from .trec_covid import get_trec_covid_abstracts
from .msmarco import convert_msmarco, extract_train_set, join_query_qrels


@click.group()
def extract():
    pass


extract.add_command(get_trec_covid_abstracts, 'trec_covid')
extract.add_command(convert_msmarco, 'msmarco')
extract.add_command(extract_train_set, 'msmarco-train')
extract.add_command(join_query_qrels, 'join-query-qrel')
