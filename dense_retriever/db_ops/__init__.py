import click
from .db_client import create_table, upload_data


@click.group()
def db_ops():
    pass


db_ops.add_command(create_table, 'create')
db_ops.add_command(upload_data, 'upload')
