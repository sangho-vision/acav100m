import json
import tempfile

import torch
import torchvision
import webdataset as wds


with open("train.json", "r") as f:
    metadata = json.load(f)


idx2class = sorted(
    {
        metadata[yid]['annotations']['label']
        for yid in metadata.keys()
    }
)

class2idx = {c: idx for idx, c in enumerate(idx2class)}


def mp4decode(data):
    with tempfile.TemporaryDirectory() as dname:
        with open(dname+"/sample.mp4", "wb") as stream:
            stream.write(data)
        frames, waveform, info = torchvision.io.read_video(dname+"/sample.mp4")
    ##############################
    # *** preprocessing code *** #
    ##############################

    return frames, waveform


def jsondecode(data):
    anno = json.loads(data)
    label = anno['annotations']['label']
    return torch.tensor(class2idx[label], dtype=torch.long)


train_data = (
    wds.Dataset("shards-train/shard-{000000..000019}.tar")
    .to_tuple("mp4", "json")
    .map_tuple(mp4decode, jsondecode)
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=10,
)

for (frames, waveform), labels in train_loader:
    print(frames.size(), waveform.size(), labels.size())
    break
