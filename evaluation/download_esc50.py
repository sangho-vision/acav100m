import wget
from pathlib import Path
from rarfile import RarFile
from zipfile import ZipFile


project_dir = Path(__file__).resolve().parent
dataset_root = project_dir / "datasets"
dataset_dir = dataset_root / "esc50"

# If you encounter a certificate error, uncomment the followig two lines.
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


def download_file(url, out=None):
    """Download a file via wget"""
    url = str(url)
    if out is not None:
        out = Path(out)
        if out.is_file():
            print(f"File already exists: {str(out)}")
        out = str(out)
    wget.download(url, out=out)


def prepare_esc50_data():
    """Download and extract ESC-50"""
    zip_filename = "esc50.zip"
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zipfile_path = dataset_root / zip_filename

    dataset_root.mkdir(parents=True, exist_ok=True)

    # Prepare ESC-50 data
    # Download ESC-50 zipfile
    print(f"Downloading data {url} to {zipfile_path}")
    download_file(url, zipfile_path)

    # Extract zipfile
    print("Extracting files...")
    zip_ref = ZipFile(zipfile_path, "r")
    zip_ref.extractall(dataset_root)
    zip_ref.close()
    dataset_root.joinpath("ESC-50-master").rename(dataset_dir)
    print("Extraction done!")
    zipfile_path.unlink()


if __name__ == "__main__":
    prepare_esc50_data()
