# Feature Extraction

## Description

We extract two feature vectors per clip using off-the-shelf feature extractors.

- Visual: SlowFast (pretrained on Kinetics-400)
- Audio: VGGish (pretrained on YouTube-8M)

## Command

1. Bundle the segmented clips into tar files.

`bash bundle.sh <path to video clip directory>/shard-000000.tar`

The `bundle.sh` script is only for demonstration purposes.
On large scale, we advise the users to use multiprocessing for bundling.

2. Run the commandline interface.

```bash
python cli.py extract --tar_path="<path to video directory>/shard-{<shard range>}.tar" \
  --out_path=<path to feature directory>"
```

e\.g\. `python cli.py extract --tar_path="data/video/shard-00000{0..4}.tar ..."`

We use [braceexpand](https://pypi.org/project/braceexpand/) for expanding shard ranges.
The above command will automatically skip a shard when there is no tar file or there already is a output pkl file.

## File Structures

### Inputs

- `shard-{shard_num}.tar`
  - per clip: `{yt-id}_{start (int)}.mp4`
- `shard-{shard_num}.json`: list of dict
  - per clip: `{'filename': FILENAME, 'id': ID, 'segment': [START, END]}`

### Outputs

- `shard-{shard_num}.pkl`: list of dict
  - per clip:
    ```{
        'filename': FILENAME, 'id': ID, 'segment': [START, END],
        'visual_features': [
          {
            'extractor_name': 'SLOWFAST_8x8_R50', 'dataset': 'kinetics-400',
            'feat_array': {'layer_0': np.ndarray, 'layer_1': np.ndarray, ...}
          },
          ...
        ],
        'audio_features': [
          {
            'extractor_name': 'VGGish', 'dataset': 'YouTube-8M',
            'feat_array': {'layer_0': np.ndarray, 'layer_1': np.ndarray, ...}
          },
          ...
        ],
        ...
        }
    ```

### Logs

```
path: {out_path}/{filename}
filename: 'log_{hostname}_{main process pid}_{timestamp}.json'
file structure:
{
 'hostname': hostname,
 'pid': pid,
 'timestamp': timestamp,
 'time': time e.g.) 2020-10-17 -,
 'shards': [shard_list]
}
```
