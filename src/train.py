import os

from registry import registry as r
from utils import group_weight

def prepare_directories(config):
    os.makedirs(os.path.join(config.train.dir, 'checkpoint'), exist_ok=True)

def train_single_epoch():
    pass

def train(config, model, loss, optimizer, scheduler):
    pass


def run(config):
    # prepare directories
    prepare_directories(config)

    # build model
    model = r.build_model(config)
    model = model.cuda()

    # build loss
    loss = r.build_loss(config)
    
    # build optimizer
    if 'no_bias_decay' in config.train and config.train.no_bias_decay:
        group_decay, group_no_decay = group_weight(model)
        params = [{'params': group_decay}, {'params': group_no_decay, 'weight_decay': 0.0}]
    else:
        params = model.parameters()    
    optimizer = r.build_optimizer(config, params=params)
    
    # build scheduler
    scheduler = r.build_scheduler(config, optimizer=optimizer)

    # train loop
    train(config=config,
          model=model,
          loss=loss,
          optimizer=optimizer,
          scheduler=scheduler)
