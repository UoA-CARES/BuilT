
import logging

from built.forward_hook import ForwardHookBase
from built.registry import Registry


@Registry.register(category='hooks')
class TweetIndexExtractionForwardHook(ForwardHookBase):
    def forward(self, inputs, model, is_train):
        outputs = model(
            input_ids=inputs['ids'],
            attention_mask=inputs['masks'],
        )

        return outputs