
import logging

from built.forward_hook import ForwardHookBase
from built.registry import Registry


@Registry.register(category='hooks')
class TweetForwardHook(ForwardHookBase):
    def forward(self, inputs, model, is_train):
        _, outputs = model(
            input_ids=inputs['ids'],
            attention_mask=inputs['masks'],
            # token_type_ids=token_type_ids
        )

        return outputs