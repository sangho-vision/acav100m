"""Meters."""
import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer

import utils.logging as logging
import utils.metrics as metrics

logger = logging.get_logger(__name__)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class ContrastMeter(object):
    """
    Measure pretraining stats for contrastive task.
    """

    def __init__(self, writer, epoch_iters, num_epochs, num_iters, cfg):
        """
        Args:
            writer (SummaryWriter): tensorboard summary writer.
            epoch_iters (int): the overall number of iterations of one epoch.
            num_epochs (int): the overall number of epochs
            num_iters (int): the overall number of iterations
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.writer = writer
        self.epoch_iters = epoch_iters
        self.NUM_EPOCHS = num_epochs
        self.NUM_ITERS = num_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.num_samples = 0
        self.lr = None
        self.acc_contrast = ScalarMeter(cfg.LOG_PERIOD)
        self.total_contrast = 0.0
        self.num_contrast = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.num_samples = 0
        self.lr = None
        self.acc_contrast.reset()
        self.total_contrast = 0.0
        self.num_contrast = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, loss, acc, lr, mb_size):
        """
        Update the current stats.
        Args:
            loss (float): loss value.
            acc (float): accuracy value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.lr = lr
        self.loss.add_value(loss)
        self.loss_total += loss * mb_size
        self.num_samples += mb_size
        self.acc_contrast.add_value(acc)
        self.total_contrast += 2 * mb_size * acc
        self.num_contrast += 2 * mb_size

    @property
    def use_plot(self):
        """
        Whether to log the stats on tensorboard.
        """
        return self.writer is not None

    def log_iter_stats(self, cur_epoch, cur_iter, global_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration in current epoch.
            global_iter (int): the number of iterations so far (one-based).
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return None
        eta_sec = self.iter_timer.seconds() * (
            self.NUM_ITERS - global_iter
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "contrast_iter",
            "global_iter": "{}/{}".format(global_iter, self.NUM_ITERS),
            "epoch": "{}/{}".format(cur_epoch + 1, self.NUM_EPOCHS),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "acc": self.acc_contrast.get_win_median(),
            "lr": self.lr,
        }

        logging.log_json_stats(stats)
        if self.use_plot:
            self.writer.add_scalar(
                "Contrast/iter_loss",
                stats["loss"],
                global_iter,
            )
            self.writer.add_scalar(
                "Contrast/iter_acc",
                stats["acc"],
                global_iter,
            )
        return stats

    def log_epoch_stats(self, cur_epoch, global_iter):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            global_iter (int): the number of iterations so far (one-based).
        """
        eta_sec = self.iter_timer.seconds() * (
            self.NUM_ITERS - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        avg_loss = self.loss_total / self.num_samples
        avg_acc = self.total_contrast / self.num_contrast
        stats = {
            "_type": "pretrain_epoch",
            "global_iter": "{}/{}".format(global_iter, self.NUM_ITERS),
            "epoch": "{}/{}".format(cur_epoch + 1, self.NUM_EPOCHS),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "loss": avg_loss,
            "acc": avg_acc,
            "lr": self.lr,
        }

        logging.log_json_stats(stats)
        if self.use_plot:
            self.writer.add_scalar(
                "Contrast/epoch_loss",
                stats["loss"],
                global_iter,
            )
            self.writer.add_scalar(
                "Contrast/epoch_acc",
                stats["acc"],
                global_iter,
            )


class ClassifyTrainMeter(object):
    """
    Measure classfication training stats.
    """

    def __init__(self, writer, epoch_iters, num_epochs, num_iters, cfg):
        """
        Args:
            writer (SummaryWriter): tensorboard summary writer.
            epoch_iters (int): the overall number of iterations of one epoch.
            num_epochs (int): the overall number of epochs
            num_iters (int): the overall number of iterations
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.writer = writer
        self.epoch_iters = epoch_iters
        self.NUM_EPOCHS = int(num_epochs)
        self.NUM_ITERS = int(num_iters)
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch accs (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Number of correctly classified examples.
        self.num_top1_acc = 0
        self.num_top5_acc = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.num_top1_acc = 0
        self.num_top5_acc = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_acc, top5_acc, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy.
            top5_acc (float): top5 accuracy.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        # Current minibatch stats
        self.mb_top1_acc.add_value(top1_acc)
        self.mb_top5_acc.add_value(top5_acc)
        # Aggregate stats
        self.num_top1_acc += top1_acc * mb_size
        self.num_top5_acc += top5_acc * mb_size

    @property
    def use_plot(self):
        """
        Whether to log the stats on tensorboard.
        """
        return self.writer is not None

    def log_iter_stats(self, cur_epoch, cur_iter, global_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration in current epoch.
            global_iter (int): the number of current iteration in training phase
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.NUM_ITERS - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "global_iter": "{}/{}".format(global_iter, self.NUM_ITERS),
            "epoch": "{}/{}".format(cur_epoch + 1, self.NUM_EPOCHS),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
        }
        stats["top1_acc"] = self.mb_top1_acc.get_win_median()
        stats["top5_acc"] = self.mb_top5_acc.get_win_median()
        logging.log_json_stats(stats)
        if self.use_plot:
            self.writer.add_scalar(
                "Train/iter_loss",
                stats["loss"],
                global_iter,
            )
            self.writer.add_scalar(
                "Train/iter_top1_acc",
                stats["top1_acc"],
                global_iter,
            )
            self.writer.add_scalar(
                "Train/iter_top5_acc",
                stats["top5_acc"],
                global_iter,
            )

    def log_epoch_stats(self, cur_epoch, global_iter):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            global_iter (int): the number of current iteration in training phase
        """
        eta_sec = self.iter_timer.seconds() * (
            self.NUM_ITERS - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "global_iter": "{}/{}".format(global_iter, self.NUM_ITERS),
            "epoch": "{}/{}".format(cur_epoch + 1, self.NUM_EPOCHS),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
        }
        top1_acc = self.num_top1_acc / self.num_samples
        top5_acc = self.num_top5_acc / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats["top1_acc"] = top1_acc
        stats["top5_acc"] = top5_acc
        stats["loss"] = avg_loss
        logging.log_json_stats(stats)
        if self.use_plot:
            self.writer.add_scalar(
                "Train/epoch_loss",
                stats["loss"],
                cur_epoch + 1,
            )
            self.writer.add_scalar(
                "Train/epoch_top1_acc",
                stats["top1_acc"],
                cur_epoch + 1,
            )
            self.writer.add_scalar(
                "Train/epoch_top5_acc",
                stats["top5_acc"],
                cur_epoch + 1,
            )


class ClassifyValMeter(object):
    """
    Measures classification validation stats.
    """

    def __init__(self, writer, epoch_iters, num_epochs, cfg):
        """
        Args:
            writer (SummaryWriter): tensorboard summary writer.
            epoch_iters (int): the overall number of iterations of one epoch.
            num_epochs (int): the overall number of epochs
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.writer = writer
        self.epoch_iters = epoch_iters
        self.NUM_EPOCHS = int(num_epochs)
        self.iter_timer = Timer()
        # Current minibatch accs (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Max accs (over the full val set).
        self.max_top1_acc = 0.0
        self.max_top5_acc = 0.0
        # Number of correctly classified examples.
        self.num_top1_acc = 0
        self.num_top5_acc = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.num_top1_acc = 0
        self.num_top5_acc = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_acc, top5_acc, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy.
            top5_acc (float): top5 accuracy.
            mb_size (int): mini batch size.
        """
        self.mb_top1_acc.add_value(top1_acc)
        self.mb_top5_acc.add_value(top5_acc)
        self.num_top1_acc += top1_acc * mb_size
        self.num_top5_acc += top5_acc * mb_size
        self.num_samples += mb_size

    @property
    def use_plot(self):
        """
        Whether to log the stats on tensorboard.
        """
        return self.writer is not None

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.epoch_iters - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self.NUM_EPOCHS),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
        }
        stats["top1_acc"] = self.mb_top1_acc.get_win_median()
        stats["top5_acc"] = self.mb_top5_acc.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self.NUM_EPOCHS),
            "time_diff": self.iter_timer.seconds(),
        }
        top1_acc = self.num_top1_acc / self.num_samples
        top5_acc = self.num_top5_acc / self.num_samples
        self.max_top1_acc = max(self.max_top1_acc, top1_acc)
        self.max_top5_acc = max(self.max_top5_acc, top5_acc)

        stats["top1_acc"] = top1_acc
        stats["top5_acc"] = top5_acc
        stats["max_top1_acc"] = self.max_top1_acc
        stats["max_top5_acc"] = self.max_top5_acc
        logging.log_json_stats(stats)
        if self.use_plot:
            self.writer.add_scalar(
                "Val/epoch_top1_acc",
                stats["top1_acc"],
                cur_epoch + 1,
            )
            self.writer.add_scalar(
                "Val/epoch_top5_acc",
                stats["top5_acc"],
                cur_epoch + 1,
            )


class ClassifyTestMeter(object):
    """
    Perform the multi-view ensemble for testing: each data point with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the data point.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_points,
        num_clips,
        num_cls,
        overall_iters,
        ensemble_method="sum",
        log_period=1,
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each data point, and calculate the metrics on
        num_points data points.
        Args:
            num_points (int): number of data points to test.
            num_clips (int): number of clips sampled from each data point for
                aggregating the final prediction for the data point.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
            log_period (int): log period.
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method
        self.log_period = log_period
        # Initialize tensors.
        self.point_preds = torch.zeros((num_points, num_cls))

        self.point_labels = torch.zeros((num_points)).long()
        self.clip_count = torch.zeros((num_points)).long()
        self.view_count = torch.zeros((num_points, num_clips)).long()
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.view_count.zero_()
        self.point_preds.zero_()
        self.point_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            point_id = int(clip_ids[ind]) // self.num_clips
            clip_id = int(clip_ids[ind]) % self.num_clips
            if self.view_count[point_id, clip_id] == 1:
                continue
            if self.point_labels[point_id].sum() > 0:
                assert torch.equal(
                    self.point_labels[point_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.point_labels[point_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.point_preds[point_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.point_preds[point_id] = torch.max(
                    self.point_preds[point_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[point_id] += 1
            self.view_count[point_id, clip_id] = 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        if (cur_iter + 1) % self.log_period != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}/{}".format(cur_iter + 1, self.overall_iters),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5), cur_epoch=None, num_epochs=None, writer=None):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        stats = {"split": "test_final"}
        if cur_epoch is not None and num_epochs is not None:
            stats['epoch'] = "{}/{}".format(cur_epoch + 1, num_epochs)
        num_topks_correct = metrics.topks_correct(
            self.point_preds, self.point_labels, ks
        )
        topks = [
            (x / self.point_preds.size(0)) * 100.0
            for x in num_topks_correct
        ]
        assert len({len(ks), len(topks)}) == 1
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                topk, prec=2
            )
        logging.log_json_stats(stats)
        if writer is not None and cur_epoch is not None and num_epochs is not None:
            for k, topk in zip(ks, topks):
                writer.add_scalar(
                    f"Test/epoch_top{k}_acc",
                    topk.item(),
                    cur_epoch + 1,
                )

        return stats
