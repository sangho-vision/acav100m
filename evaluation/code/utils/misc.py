import math
from datetime import datetime

import utils.logging as logging

logger = logging.get_logger(__name__)


def check_nan_losses(loss):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def is_eval_epoch(eval_period, cur_epoch, num_epochs):
    """
    Determine if the model should be evaluated at the current epoch.
    """
    return (
        cur_epoch + 1
    ) % eval_period == 0 or cur_epoch + 1 == num_epochs
