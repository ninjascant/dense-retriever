import os
import json
import pickle
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


def write_jsonl_file(dict_list, out_file):
    json_strings = [json.dumps(obj) for obj in dict_list]
    with open(out_file, 'w') as outfile:
        outfile.write('\n'.join(json_strings) + '\n')


def read_qrel_file(file_path):
    qrels = pd.read_csv(file_path, sep=' ', header=None)
    qrels.columns = ['qid', 'none1', 'doc_id', 'none2']
    qrels = qrels[['qid', 'doc_id']]
    return qrels


def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def write_pickle_file(file_path, obj):
    with open(file_path, 'wb') as f:
        return pickle.dump(obj, f)


def load_json_file(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)
