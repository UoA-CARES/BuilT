
import logging
from built.forward_hook import ForwardHookBase
from built.registry import Registry


@Registry.register(category='hooks')
class TweetForwardHook(ForwardHookBase):
    def __call__(self, model, inputs, targets=None, data=None, is_train=False):
        logging.debug("Tweet forward hook is called")

        d = inputs 

        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']        
        targets = d['sentiment_tar']

        model.zero_grad()
        _, outputs = model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        return outputs