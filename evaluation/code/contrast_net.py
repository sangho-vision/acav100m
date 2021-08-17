import os
import math
import random
import pprint
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import models.optimizer as optim
import utils.distributed as du
import utils.checkpoint as cu
import utils.logging as logging
import utils.misc as misc
from data import loader
from models import build_model
from utils.meters import ContrastMeter


logger = logging.get_logger(__name__)


def contrast(cfg):
    """
    Pretrain an audio-visual model in a cross-modal contrastive manner.
    Args:
        cfg (CfgNode) : configs. Details can be found in
            config.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    if cfg.RNG_SEED != -1:
        random.seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Setup logging format.
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    logging.setup_logging(os.path.join(cfg.LOG_DIR, f"log-{timestamp}.txt"))

    # Print config.
    logger.info("Contrastive Task with config:")
    logger.info(pprint.pformat(cfg))

    # Audio-visual model for pretraining.
    model = build_model(cfg)
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the audio-visual pretrain loader.
    pretrain_loader = loader.construct_loader(cfg, 'pretrain')

    num_batches_per_epoch = len(pretrain_loader)
    # Priority : MAX_EPOCH > NUM_STEPS
    # Priority : WARMUP_STEPS > WARMUP_EPOCHS > WARMUP_PROPORTION
    if cfg.SOLVER.MAX_EPOCH != -1:
        num_optimizer_epochs = cfg.SOLVER.MAX_EPOCH
        num_optimizer_steps = (
            num_optimizer_epochs * num_batches_per_epoch
        )
        if cfg.SOLVER.WARMUP_STEPS != -1:
            num_warmup_steps = cfg.SOLVER.WARMUP_STEPS
            num_warmup_epochs = num_warmup_steps / num_batches_per_epoch
        elif cfg.SOLVER.WARMUP_EPOCHS != -1:
            num_warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
            num_warmup_steps = (
                num_warmup_epochs * num_batches_per_epoch
            )
        else:
            num_warmup_steps = (
                num_optimizer_steps * cfg.SOLVER.WARMUP_PROPORTION
            )
            num_warmup_epochs = num_warmup_steps / num_batches_per_epoch
        num_epochs = cfg.SOLVER.MAX_EPOCH
        num_steps = num_epochs * num_batches_per_epoch
    else:
        num_optimizer_steps = cfg.SOLVER.NUM_STEPS
        num_optimizer_epochs = num_optimizer_steps / num_batches_per_epoch
        if cfg.SOLVER.WARMUP_STEPS != -1:
            num_warmup_steps = cfg.SOLVER.WARMUP_STEPS
            num_warmup_epochs = num_warmup_steps / num_batches_per_epoch
        elif cfg.SOLVER.WARMUP_EPOCHS != -1:
            num_warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
            num_warmup_steps = (
                num_warmup_epochs * num_batches_per_epoch
            )
        else:
            num_warmup_steps = (
                num_optimizer_steps * cfg.SOLVER.WARMUP_PROPORTION
            )
            num_warmup_epochs = num_warmup_steps / num_batches_per_epoch
        num_steps = cfg.SOLVER.NUM_STEPS
        num_epochs = math.ceil(num_steps / num_batches_per_epoch)

    start_epoch = 0
    global_step = 0

    if cfg.PRETRAIN.PREEMPTIBLE:
        pretrain_checkpoint_file_path = os.path.join(
            cfg.SAVE_DIR,
            "epoch_latest.pyth",
        )
    else:
        pretrain_checkpoint_file_path = cfg.PRETRAIN.CHECKPOINT_FILE_PATH

    if os.path.isfile(pretrain_checkpoint_file_path) and 'epoch' in pretrain_checkpoint_file_path:
        logger.info(
            "=> loading checkpoint '{}'".format(
                pretrain_checkpoint_file_path
            )
        )
        # Load the checkpoint on CPU to avoid GPU mem spike.
        checkpoint = torch.load(
            pretrain_checkpoint_file_path, map_location='cpu'
        )
        cu.load_checkpoint(
            model,
            checkpoint['state_dict'],
            cfg.NUM_GPUS > 1,
        )
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        global_step = checkpoint['epoch'] * len(pretrain_loader)
        logger.info(
            "=> loaded checkpoint '{}'".format(
                pretrain_checkpoint_file_path,
            )
        )

    writer = None
    if du.is_master_proc():
        writer = SummaryWriter(cfg.LOG_DIR)

    # Create meters
    pretrain_meter = ContrastMeter(
        writer,
        len(pretrain_loader),
        num_epochs,
        num_steps,
        cfg,
    )

    # Perform the pretraining loop.
    logger.info("Start epoch: {}".format(start_epoch+1))

    for cur_epoch in range(start_epoch, num_epochs):
        # Shuffle the dataset.
        loader.shuffle_dataset(pretrain_loader, cur_epoch)
        # Pretrain for one epoch.
        global_step = contrast_epoch(
            pretrain_loader,
            model,
            optimizer,
            pretrain_meter,
            cur_epoch,
            global_step,
            num_steps,
            num_optimizer_steps,
            num_warmup_steps,
            cfg,
        )

        sd = \
            model.module.state_dict() if cfg.NUM_GPUS > 1 else \
            model.state_dict()

        ckpt = {
            'epoch': cur_epoch + 1,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
        }

        if cfg.PRETRAIN.PREEMPTIBLE and du.get_rank() == 0:
            cu.save_checkpoint(
                ckpt, filename=os.path.join(cfg.SAVE_DIR, "epoch_latest.pyth")
            )

        if (cur_epoch + 1) % cfg.PRETRAIN.SAVE_EVERY_EPOCH == 0 and du.get_rank() == 0:
            cu.save_checkpoint(
                ckpt,
                filename=os.path.join(cfg.SAVE_DIR, f"epoch{cur_epoch+1}.pyth")
            )

        if global_step == num_steps:
            break


def contrast_epoch(
    pretrain_loader,
    model,
    optimizer,
    pretrain_meter,
    cur_epoch,
    global_step,
    num_steps,
    num_optimizer_steps,
    num_warmup_steps,
    cfg,
):
    # Switch to train mode.
    model.train()
    pretrain_meter.iter_tic()

    for cur_step, (visual_clip, audio_clip) in enumerate(pretrain_loader):
        global_step += 1
        for i in range(len(visual_clip)):
            visual_clip[i] = visual_clip[i].cuda(non_blocking=True)
        audio_clip = audio_clip.cuda(non_blocking=True)
        loss, acc = model(visual_clip, audio_clip)

        # Check Nan Loss.
        misc.check_nan_losses(loss)

        loss.backward()
        lr = optim.get_lr(
            cfg.SOLVER.LR_POLICY,
            cfg.SOLVER.BASE_LR,
            cfg.SOLVER.WARMUP_START_LR,
            global_step,
            num_optimizer_steps,
            num_warmup_steps,
        )
        optim.set_lr(optimizer, lr)
        optimizer.step()
        for p in model.parameters():
            p.grad = None

        # Gather all the stats across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, acc = du.all_reduce([loss, acc])

        loss = loss.item()
        acc = acc.item()

        pretrain_meter.iter_toc()
        # Update and log stats.
        pretrain_meter.update_stats(
            loss,
            acc,
            lr,
            audio_clip.size(0) * du.get_world_size()
        )

        pretrain_meter.log_iter_stats(
            cur_epoch, cur_step, global_step,
        )

        sd = \
            model.module.state_dict() if cfg.NUM_GPUS > 1 else \
            model.state_dict()
        ckpt = {
            'step': global_step,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
        }

        if cfg.PRETRAIN.PREEMPTIBLE and global_step % cfg.PRETRAIN.SAVE_PERIOD == 0 and du.get_rank() == 0:
            path = f"step_latest.pyth"
            cu.save_checkpoint(
                ckpt,
                filename=os.path.join(cfg.SAVE_DIR, path)
            )

        if global_step % cfg.LOG_PERIOD == 0:
            logger.info("PROGRESS: {}%".format(round(100*(global_step) / num_steps, 4)))
            logger.info("EVALERR: {}%".format(loss))

        if global_step == num_steps and (cur_step + 1) != len(pretrain_loader):
            return global_step

        pretrain_meter.iter_tic()

    # Log epoch stats.
    pretrain_meter.log_epoch_stats(cur_epoch, global_step)
    pretrain_meter.reset()

    return global_step
