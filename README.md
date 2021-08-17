# ACAV100M: Automatic Curation of Large-Scale Datasets for Audio-Visual Video Representation Learning

This repository contains the code for our ICCV 2021 paper:

**ACAV100M: Automatic Curation of Large-Scale Datasets for Audio-Visual Video Representation Learning** <br>
Sangho Lee\*, Jiwan Chung\*, Youngjae Yu, Gunhee Kim, Thomas Breuel, Gal Chechik, Yale Song (\*: equal contribution) <br>
[[paper]](https://arxiv.org/abs/2101.10803)

```bibtex
@inproceedings{lee2021acav100m,
    title="{ACAV100M: Automatic Curation of Large-Scale Datasets for Audio-Visual Video Representation Learning}",
    author={Sangho Lee and Jiwan Chung and Youngjae Yu and Gunhee Kim and Thomas Breuel and Gal Chechik and Yale Song},
    booktitle={ICCV},
    year=2021
}
```

## System Requirements

- Python >= 3.8.5
- FFMpeg 4.3.1

## Installation

1. Install PyTorch 1.6.0, torchvision 0.7.0 and torchaudio 0.6.0 for your environment.
Follow the instructions in
[HERE](https://pytorch.org/get-started/previous-versions/).

2. Install the other required packages.

```bash
pip install -r requirements.txt
python -m nltk.downloader 'punkt'
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/<cuda version>/torch1.6/index.html
pip install git+https://github.com/jiwanchung/slowfast
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.6.0+<cuda version>.html
```

e\.g\. Replace `<cuda version>` with `cu102` for CUDA 10.2.

## Input File Structure

1. Create the data directory

```
mkdir data
```

2. Prepare the input file.

`data/metadata.tsv` should be structured as follows.
We provide an example input file in `examples/metadata.tsv`

```
YOUTUBE_ID\t{"LatestDAFeature": {"Title": TITLE, "Description": DESCRIPTION, "YouTubeCategory": YOUTUBE_CATEGORY, "VideoLength": VIDEO_LENGTH}, "MediaVersionList": [{"Duration": DURATION}]}
```

## Data Curation Pipeline

### One-Liner

`bash ./run.sh`

To enable GPU computation, modify the `CUDA_VISIBLE_DEVICES` environment variable accordingly.
For example, run the above command as `export CUDA_VISIBLE_DEVICES=2,3; bash ./run.sh`.

### Step-by-Step

1. Filter the videos with metadata.

```bash
bash ./metadata_filtering/code/run.sh
```

The above command will build the `data/filtered.tsv` file.

2. Download the actual video files from youtube.

```bash
bash ./video_download/code/run.sh
```

Although we provide a simple download script,
we recommend more scalable solutions for downloading large-scale data.

The above command will download the files to `data/videos/raw` directory.

3. Segment the videos into 10-second clips.

```bash
bash ./clip_segmentation/code/run.sh
```

The above command will save the segmented clips to `data/videos` directory.

4. Extract features from the clips.

```bash
bash ./feature_extraction/code/run.sh
```

The above command will save the extracted features to `data/features` directory.

This step requires GPU for faster computation.

5. Perform clustering with the extracted features.

```bash
bash ./clustering/code/run.sh
```

The above command will save the extracted features to `data/clusters` directory.

This step requires GPU for faster computation.

6. Select subset with high audio-visual correspondence using the clustering results.

```bash
bash ./subset_selection/code/run.sh
```

The above command will save the selected clip indices to `data/datasets` directory.

This step requires GPU for faster computation.

The final output should be saved in the `data/output.csv` file.

## Output File Structure

`output.csv` is structured as follows.
We provide an example output file at `examples/output.csv`.

```
# SHARD_NAME,FILENAME,YOUTUBE_ID,SEGMENT
shard-000009,qpxektwhzra_292.mp4,qpxektwhzra,"[292.3329999997, 302.3329999997]"
```

## Evaluation

Instructions on downstream evaluation are provided in [Evaluation](evaluation/README.md).

## Correspondence Retrieval

Instructions on correspondence retrieval experiments are provided in [Correspondence Retrieval](correspondence_retrieval/README.md).
