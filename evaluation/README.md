# Evaluation on Downstream Tasks


## Pretrain

To run pretraining on our dataset, ACAV100M:

```bash
cd code
python run_net.py \
    --cfg_file configs/acav/config.yaml \
    --configuration acav100m \
    PRETRAIN.DATASET_SIZE 100000000 \
    DATASET_DIR <path to dataset directory>
```


## Linear Evaluation

### Download Data

Download UCF101, ESC-50 and Kinetics-Sounds. You may encounter a certificate error downloading UCF101 or ESC-50. Please see the comments in `download_ucf101.py` or `download_esc50.py`.

```bash
python download_ucf101.py
python download_esc50.py
python download_ks.py
```

### Run Experiments

To run experiments on visual, audio and audio-visual datasets:

UCF101 (split: 1, 2, or 3)
```bash
cd code
python run_net.py \
    --cfg_file configs/ucf101/config.yaml \
    --configuration ucf101-<split> \
    --pretrain_checkpoint_path <path to pretraining checkpoint> \
    TRAIN.DATASET_SPLIT <split>
    TEST.DATASET_SPLIT <split>
```

ESC-50 (split: 1, 2, 3, 4 or 5)
```bash
cd code
python run_net.py \
    --cfg_file configs/esc50/config.yaml \
    --configuration esc50-<split> \
    --pretrain_checkpoint_path <path to pretraining checkpoint> \
    TRAIN.DATASET_SPLIT <split>
    TEST.DATASET_SPLIT <split>
```

Kinetics-Sounds
```bash
cd code
python run_net.py \
    --cfg_file configs/kinetics-sounds/config.yaml \
    --configuration kinetics-sounds \
    --pretrain_checkpoint_path <path to pretraining checkpoint> \
```

### Pretrained Model

We provide the checkpoint for the model pretrained on our dataset, ACAV100M.
The checkpoint will be saved in `code/checkpoints`.
```bash
python download_checkpoint.py
```

| Dataset | Top-1 Accuracy | Top-5 Accuracy |
| ------- | -------------- | -------------- |
| UCF101 | 86.10 | 97.94 |
| ESC-50 | 86.95 | 97.45 |
| Kinetics-Sounds | 75.42 | 95.88 |


## Acknowledgments
This source code is based on [PySlowFast](https://github.com/facebookresearch/SlowFast).
