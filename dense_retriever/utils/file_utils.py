import os
import json
import tarfile
import pandas as pd


def zip_dir(dir_path, archive_path):
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))


def unzip_arch(archive_path):
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall()


def read_jsonl_file(file_path):
    with open(file_path) as jsonl_file:
        lines = jsonl_file.readlines()
    data = [json.loads(line) for line in lines]
    return data


def read_qrel_file(file_path):
    qrels = pd.read_csv(file_path, sep=' ', header=None)
    qrels.columns = ['qid', 'none1', 'doc_id', 'none2']
    qrels = qrels[['qid', 'doc_id']]
    return qrels
