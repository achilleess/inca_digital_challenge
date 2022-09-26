import os
import os.path as osp
import sys
import argparse
from collections import defaultdict

import pandas as pd
import torch
from mmcv import Config

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
    parser.add_argument("weights_path", type=str)
    parser.add_argument("save_file_path", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("training_fold", type=str)
    args = parser.parse_args()
    return args


def create_save_file(config, save_path):
    row_data = pd.read_csv(config.row_data_path)
    data = pd.read_csv(config.data_path)

    texts = list(row_data.groupby('Text').groups)
    res_df = pd.DataFrame({'Text': texts})

    txt_to_fold = defaultdict(lambda: -2)
    txt_to_agr_rate = defaultdict(lambda: -1)
    for i in range(len(data)):
        row = data.iloc[i]
        txt_to_fold[row.Text] = row.fold
        txt_to_agr_rate[row.Text] = row.agreement_rate
    
    res_df['fold'] = res_df['Text'].map(txt_to_fold)
    res_df['agr_rate'] = res_df['Text'].map(txt_to_agr_rate)
    res_df.to_csv(save_path, index=False)
    return res_df


if __name__=='__main__':
    args = parse_args()

    config = Config.fromfile(args.config_path)
    config.train_procedure.use_wandb = False

    config.device = torch.device(config.train_procedure['device_name'])

    if not osp.isfile(args.save_file_path):
        data = create_save_file(config, args.save_file_path)
    else:
        data = pd.read_csv(args.save_file_path)

    print(data)
    df = data.copy()
    fold_to_train = 0
    df['fold'] = list([fold_to_train] * len(df))
    config.fold_to_train = fold_to_train

    val_loader = get_loaders(config, df, val_only=True)
    
    model = build_model(config.model)
    model.load_state_dict(torch.load(args.weights_path)['model'])
    model.cuda().eval()

    trainer = DefaultTrainerFP16(
        model=model,
        config=config
    )

    indexes, outputs = trainer.get_predictions(
        loader=val_loader
    )

    outputs = torch.sigmoid(outputs.float()).detach().cpu().numpy()
    indexes = indexes.detach().cpu().numpy()

    ziped_out = list(zip(outputs, indexes))
    ziped_out[10], ziped_out[0] = ziped_out[0], ziped_out[10]
    ziped_out = sorted(ziped_out, key=lambda x: x[1])

    preds = [float(pred) for pred, idx in ziped_out]

    model_name = args.model_name + f'_fold_{args.training_fold}'

    data[model_name] = preds
    print(data)
    data.to_csv(args.save_file_path, index=False)

