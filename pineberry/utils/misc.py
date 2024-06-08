#!/usr/bin/env python3

from mmengine import MessageHub


def get_curr_iter_info(epoch: bool = True):
    """Get current iteration and max iterations from the message hub."""
    message_hub = MessageHub.get_current_instance()

    if epoch:
        curr_step = message_hub.get_info("epoch")
        max_steps = message_hub.get_info("max_epochs")
    else:
        curr_step = message_hub.get_info("iter")
        max_steps = message_hub.get_info("max_iters")
    return curr_step, max_steps
