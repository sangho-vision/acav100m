import os

import torch

import utils.multiprocessing as mpu
from config import get_cfg
from contrast_net import contrast
from classify_net import classify


RUN_DICT = {
    "Contrast": contrast,
    "VisualClassify": classify,
    "AudioClassify": classify,
    "MultimodalClassify": classify,
}


def main():
    """
    Main function to spawn the train and test process.
    """
    cfg = get_cfg()

    run = RUN_DICT[cfg.MODEL.TASK]
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                run,
                cfg.DIST_INIT_METHOD,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=False,
        )
    else:
        run(cfg=cfg)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()

