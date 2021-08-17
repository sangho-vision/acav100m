# Correspondence Retrieval Experiments

## System Requirements

- [faiss](https://github.com/facebookresearch/faiss) == 1.6.3

## Installation

1. For comparing sgd-based clustering against default clustering, we need a fast clustering package (faiss).

```bash
conda install -c pytorch faiss-gpu
```

## Command

1. Download the correspondence pair datasets.

```
python ./download_datasets.py
```

We provide google drive link to the extracted features of the datasets.
The above script downloads all available datasets.
Alternatively, you can build the necessary dataset yourself by running step 2.

2. Run a dummy experiment.

```bash
cd code
python cli.py run
```

The above script will download original datasets for integrity check.
Also, it will extract necessary features if they are not downloaded already in step 1.

3. Run all experiments.

```bash
cd code
bash ./run.sh
```

To enable GPU computation, modify the `CUDA_VISIBLE_DEVICES` environment variable accordingly.
For example, run the above command as `export CUDA_VISIBLE_DEVICES=2,3; bash ./run.sh`.

3. Check the output at the `../data/correspondence_retrieval/output` directory.
