import zipfile
import argparse
import requests
from tqdm import tqdm
from pathlib import Path


project_dir = Path(__file__).resolve().parent
root = project_dir.parent / 'data' / 'correspondence_retrieval'

FILE_ID = {
    "pair_data": "1CMQiKJqKK1VJme0T04HbCoRCAXcMY9Pc",
}


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def download_datasets(dataset, file_id):
    """Download the Datasets"""
    dataset_filename = f"{dataset}.zip"
    dataset_path = root / dataset_filename

    root.mkdir(parents=True, exist_ok=True)
    # Download the datasets
    print(f"Downloading the datasets to {dataset_path}")
    download_file_from_google_drive(file_id, str(dataset_path))
    print(f"Successfully downloaded {dataset_path}")
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall()


if __name__ == "__main__":
    name = 'pair_data'
    download_checkpoint(name, FILE_ID[name])
