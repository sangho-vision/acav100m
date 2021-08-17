"""Learning rate policy."""

import math


def get_lr(
    policy,
    base_lr,
    warmup_start_lr,
    global_step,
    num_optimizer_steps,
    num_warmup_steps,
):
    """
    Retrieve the learning rate of the current step with the option to perform
    warm up in the beginning of the training stage.
    Args:
        policy (string): learning rate policy.
        base_lr (float): base learning rate
        warmup_start_lr (float): warmup start learning rate
        global_step (int): current step
        num_optimizer_steps (int): the number of total training steps.
        num_warmup_steps (int): the number of total warmup steps.
    """
    if policy == "linear":
        alpha = lr_func_linear(global_step, num_optimizer_steps, num_warmup_steps)
        lr = base_lr * alpha
    elif policy == "cosine":
        if global_step < num_warmup_steps:
            alpha = lr_func_linear(global_step, num_optimizer_steps, num_warmup_steps)
            lr = warmup_start_lr + (base_lr - warmup_start_lr) * alpha
        else:
            lr = lr_func_cosine(base_lr, global_step - num_warmup_steps, num_optimizer_steps - num_warmup_steps)
    elif policy == "constant":
        lr = base_lr
    else:
        raise NotImplementedError(
            "Does not support {} learning policy".format(policy)
        )
    return lr


def lr_func_linear(current_step, num_training_steps, num_warmup_steps):
    """
    Retrieve the learning rate scale using the linear learning rate schedule.
    """
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def lr_func_cosine(base_lr, cur_epoch, num_optimizer_epochs):
    """
    Retrieve the learning rate of the current step using the cosine learning
    rate schedule.
    """
    return (
        base_lr
        * (math.cos(math.pi * cur_epoch / num_optimizer_epochs) + 1.0)
        * 0.5
    )
