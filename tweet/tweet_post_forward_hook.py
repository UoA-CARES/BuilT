
import logging

from built.forward_hook import PostForwardHookBase
from built.registry import Registry


@Registry.register(category='hooks')
class TweetPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs, inputs=None, targets=None, data=None, is_train=False):
        # logging.info("Default post forward hook is called")
        return outputs