import os
import json
import tarfile


def zip_dir(dir_path, archive_path):
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))


def read_jsonl_file(file_path):
    with open(file_path) as jsonl_file:
        lines = jsonl_file.readlines()
    data = [json.loads(line) for line in lines]
    return data
