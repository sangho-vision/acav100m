import json
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-p', '--clip-path', default=None, type=str)
args = parser.parse_args()

path = Path(args.clip_path)
clips = path.glob('*.mp4')
meta = []
for clip in clips:
    filename = clip.name
    _id = filename[:11]
    start = int(Path(filename[12:]).stem)
    meta.append({'filename': filename, 'id': _id, 'segment': [start, start + 10]})

with open(path.parent / 'shard-000000.json', 'w') as f:
    json.dump(meta, f)
