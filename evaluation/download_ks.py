import requests
from tqdm import tqdm
from pathlib import Path
from zipfile import ZipFile


project_dir = Path(__file__).resolve().parent
dataset_root = project_dir / "datasets"
dataset_dir = dataset_root / "kinetics-sounds"


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


def prepare_ks_data():
    """Download and extract Kinetics-Sounds"""
    zip_filename = "kinetics-sounds.zip"
    file_id = "1KYz08ikUemlqa6Gc0EkA31OH2GuSJnTS"
    zipfile_path = dataset_root / zip_filename

    dataset_root.mkdir(parents=True, exist_ok=True)
    # Prepare Kinetics-Sounds data
    # Download Kinetics-Sounds zipfile
    print(f"Downloading data to {zipfile_path}")
    download_file_from_google_drive(file_id, str(zipfile_path))
    print(f"Successfully downloaded {zipfile_path}")

    '''
    # Extract zipfile
    print("Extracting files...")
    zip_ref = ZipFile(zipfile_path, "r")
    zip_ref.extractall(dataset_root)
    zip_ref.close()
    print("Extraction done!")
    zipfile_path.unlink()
    '''


if __name__ == "__main__":
    prepare_ks_data()
