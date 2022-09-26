import os
import os.path as osp
import sys
import argparse

import torch
import torch.optim as optims
from mmcv import Config
import pandas as pd

sys.path.insert(0, osp.join(
    '/'.join(osp.dirname(__file__).split('/')[:-1])
))

from bot_detector.dataset import get_loaders
from bot_detector.models.build_model import build_model
from bot_detector.losses.build_loss import build_loss
from bot_detector.training_procedure import DefaultTrainer, DefaultTrainerFP16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("fold_to_train", type=str)
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()

    config = Config.fromfile(args.config_path)
    config.fold_to_train = args.fold_to_train

    config.device = torch.device(config.train_procedure['device_name'])

    if not osp.isdir('weights'):
        os.mkdir('weights')

    df = pd.read_csv(config.data_path)
    train_loader, val_loader = get_loaders(config, df)
    
    model = build_model(config.model)

    losses = [build_loss(loss_config) for loss_config in config.losses]
    metrics = [build_loss(metric_config) for metric_config in config.metrics]

    optimizer_cls = getattr(optims, config.optimizer.pop('type'))
    optimizer = optimizer_cls(params=model.parameters(), **config.optimizer)

    scheduler = optims.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.scheduler.max_lr,
        total_steps=config.train_procedure.epochs * len(train_loader),
        div_factor=config.scheduler.max_lr / config.scheduler.start_lr,
        final_div_factor=config.scheduler.start_lr / config.scheduler.min_lr,
        pct_start=config.scheduler.warmup_epochs / config.train_procedure.epochs
    )

    trainer = DefaultTrainerFP16(
        model=model,
        config=config,
        metrics=metrics,
        optimizer=optimizer,
        loss_functions=losses,
        scheduler=scheduler,
    )

    trainer.run(
        train_loader=train_loader,
        val_loader=val_loader
    )