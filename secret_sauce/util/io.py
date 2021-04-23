import torchaudio
from google.cloud import storage

# impl for torchaudio 0.8.0+
# def get_duration_sec(song_path: str) -> float:
#     info = torchaudio.info(song_path)
#     return info.num_frames / info.sample_rate


def get_duration_sec(song_path: str) -> float:
    sample_info, encoding_info = torchaudio.info(song_path)

    return (sample_info.length // sample_info.channels) / sample_info.rate


from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
