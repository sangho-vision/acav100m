import wget
from pathlib import Path
from rarfile import RarFile
from zipfile import ZipFile


project_dir = Path(__file__).resolve().parent
dataset_root = project_dir / "datasets"
dataset_dir = dataset_root / "ucf101"
split_dir = dataset_dir / "splits"

# If you encounter a certificate error, uncomment the followig two lines.
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


def download_file(url, out=None):
    url = str(url)
    if out is not None:
        out = Path(out)
        if out.is_file():
            print(f"File already exists: {str(out)}")
        out = str(out)
    wget.download(url, out=out)


def prepare_ucf101_data():
    """Download and extract UCF101 & split meta data"""
    rar_filename = "ucf101.rar"
    url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    rarfile_path = dataset_root / rar_filename
    zip_filename = "ucf101_splits.zip"
    split_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
    zipfile_path = dataset_root / zip_filename

    dataset_root.mkdir(parents=True, exist_ok=True)

    # Prepare UCF101 data
    # Download UCF101 rarfile
    print(f"Downloading data {url} to {rarfile_path}")
    download_file(url, rarfile_path)
    print(f"Successfully downloaded {rarfile_path}")

    # Extract rarfile
    print("Extracting videos...")
    rar_ref = RarFile(rarfile_path, "r")
    rar_ref.extractall(dataset_root)
    rar_ref.close()
    dataset_root.joinpath("UCF-101").rename(dataset_dir)
    print("Extraction done!")
    rarfile_path.unlink()

    # Prepare UCF101 split meta data
    # Download split data zipfile
    print(f"Downloading split meta data {split_url} to {zipfile_path}")
    download_file(split_url, zipfile_path)
    print(f"Successfully downloaded {zipfile_path}")

    # Extract zipfile
    print("Extracting split meta data...")
    zip_ref = ZipFile(zipfile_path, "r")
    zip_ref.extractall(dataset_dir)
    zip_ref.close()
    dataset_dir.joinpath("ucfTrainTestlist").rename(split_dir)
    print("Extraction done!")
    zipfile_path.unlink()


if __name__ == "__main__":
    prepare_ucf101_data()
