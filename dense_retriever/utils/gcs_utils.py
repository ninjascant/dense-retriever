from loguru import logger
from google.cloud import storage


def _get_bucket(bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket


def upload_file_to_gcs(bucket_name, source_path, dst_path):
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(dst_path)
    logger.info(f'Uploading file to gs://{bucket_name}/{dst_path}')
    blob.upload_from_filename(source_path)


def download_file_from_gcs(bucket_name, source_path, dst_path):
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(source_path)
    blob.download_to_filename(dst_path)
