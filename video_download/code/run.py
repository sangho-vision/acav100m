import argparse
from pathlib import Path

from tqdm import tqdm
import youtube_dl


def parse_args():
    parser = argparse.ArgumentParser(description='simple youtube downloadere')
    parser.add_argument('-p', '--data-path', type=str, required=True, help='Input Directory')
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='Output Directory')
    args = parser.parse_args()
    return args


def load_data(path):
    urls = {}
    with open(path, 'r') as f:
        for line in f:
            url, _ = line.split('\t')
            vid = url[-11:]
            urls[vid] = url
    return urls


def download(urls, output_dir):
    ydl_opts = {
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'merge_output_format': 'mp4',
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        for vid, url in tqdm(urls.items(), total=len(urls)):
            if not (output_dir / f'{vid}.mp4').is_file():
                try:
                    ydl.download([url])
                except youtube_dl.utils.DownloadError as e:
                    print(e)


def main():
    args = parse_args()
    data = load_data(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    download(data, output_dir)


if __name__ == '__main__':
    main()
