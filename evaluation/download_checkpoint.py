import requests
from tqdm import tqdm
from pathlib import Path


project_dir = Path(__file__).resolve().parent
checkpoint_root = project_dir / "code" / "checkpoints"

FILE_ID = {
    "acav100m": "1OxSj_jRmQpVjNmuyjhGdEcuogRCfo9Rk",
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


def download_checkpoint(dataset, file_id):
    """Download the checkpoint"""
    ckpt_filename = f"{dataset}_checkpoint.pyth"
    ckpt_path = checkpoint_root / ckpt_filename

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    # Download the checkpoint
    print(f"Downloading the checkpoint to {ckpt_path}")
    download_file_from_google_drive(file_id, str(ckpt_path))
    print(f"Successfully downloaded {ckpt_path}")

if __name__ == "__main__":
    name = 'acav100m'
    download_checkpoint(name, FILE_ID[name])
