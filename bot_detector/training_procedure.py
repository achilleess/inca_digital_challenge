import os.path as osp
import sys
import collections

import torch
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(osp.join(osp.dirname(__file__), '../'))

from bot_detector.stat import StatKeeper


class DefaultTrainer():
    def __init__(self, model, config, optimizer=None, metric_function=None,
                    loss_functions=None, scheduler=None, metrics=None):
        self.loss_functions = loss_functions
        self.optimizer = optimizer
        self.model = model
        self.scheduler = scheduler
        self.device = config.device
        self.epochs = config.train_procedure.epochs
        self.accum_grad_steps = config.train_procedure.accum_grad_steps
        self.exp_name = config.exp_name
        self.metric_function = metric_function
        self.config = config
        self.val_step = config.train_procedure.val_step
        self.metrics = metrics

        self.use_wandb = config.train_procedure.use_wandb
        if self.use_wandb:
            wandb.init(
                project='IncaChallenge',
                name=self.exp_name,
                config=config
            )
        self.train_step = 1
        self.epoch_num = 1
        self.model.cuda()#.to(self.device)

    def to_device(self, container):
        for n, v in container.items():
            if not isinstance(v, list):
                container[n] = v.to(self.device)

    def trainer_forward(self, container):
        self.model(container)

    def preprocessing_step(self, container):
        pass

    def calc_loss(self, container):
        container['losses'] = {}
        total_loss = 0
        for loss_foo in self.loss_functions:
            loss = loss_foo(container)
            container['losses'][loss_foo.loss_name] = loss
            total_loss = total_loss + loss * loss_foo.loss_weight
        container['losses']['total_loss'] = total_loss

    def optimizer_step(self, container):
        total_loss = container['losses']['total_loss']
        (total_loss / self.accum_grad_steps).backward()

        if self.train_step % self.accum_grad_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.train_step += 1

    def scheduler_step(self):
        if not self.scheduler is None:
            self.scheduler.step()
    
    def run_epoch(self, loader, epoch_num, train=True):
        print('\n')
        self.model.train()
        desc = 'TRAIN Epoch {}/{}'

        progressbar = tqdm(
            loader,
            bar_format='{l_bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}',
            desc=desc.format(epoch_num + 1, self.epochs)
        )

        stat_keeper = StatKeeper()
        for num_batch, container in enumerate(progressbar):
            self.to_device(container)
            self.preprocessing_step(container)
            self.trainer_forward(container)
            self.calc_loss(container)

            for metric in self.metrics:
                metric(container)
            
            if train:
                self.optimizer_step(container)
                self.scheduler_step()

            stat_keeper.step(container)
            progressbar.set_postfix(**stat_keeper.get_stat())
        
        metric_dict = self.log_metrics('train', epoch_num, loader)
        
        if self.use_wandb:
            stat = stat_keeper.get_stat()
            stat_postfix = '_train'
            stat = {i + stat_postfix: j for i, j in stat.items()}

            wandb.log(stat, step=epoch_num)

            for param_group in self.optimizer.param_groups:
                wandb.log({'lr': param_group['lr']}, step=epoch_num)
        return stat_keeper.get_stat()['total_loss_avg']
    
    def save_checkpoint(self, path):
        torch.save({
            'model': self.model.state_dict(),
            #'optimizer': self.optimizer.state_dict()
        }, path)
    
    def log_metrics(self, tag, epoch_num, loader):
        ret_dict = {}
        for metric in self.metrics:
            metric_wise = metric.calc_metric()
            ret_dict.update(metric_wise)
            metric.reset()

            if self.use_wandb:
                stat_postfix = '_' + tag
                for i, j in metric_wise.items():
                    wandb.log(
                        {i + stat_postfix: j},
                        step=epoch_num
                    )

                for param_group in self.optimizer.param_groups:
                    wandb.log({'lr': param_group['lr']}, step=epoch_num)
        return ret_dict
    
    def run_eval(self, loader, epoch_num):
        print('\n')
        self.model.eval()

        progressbar = tqdm(
            loader,
            desc= 'VAL Epoch {}/{}'.format(epoch_num + 1, self.epochs)
        )
        
        for num_batch, container in enumerate(progressbar):
            self.to_device(container)
            self.preprocessing_step(container)
                                
            with torch.no_grad(), torch.cuda.amp.autocast():
                self.trainer_forward(container)

            for metric in self.metrics:
                metric(container)
            torch.cuda.empty_cache()

            #if num_batch == 100:
            #    break
        
        ret_dict = self.log_metrics('val', epoch_num, loader)
        return ret_dict['F1_score']
    
    def get_predictions(self, loader):
        self.model.eval()

        indexes = []
        outputs = []

        progressbar = tqdm(loader, desc= "Inference")
        for num_batch, container in enumerate(progressbar):
            self.to_device(container)
            self.preprocessing_step(container)
                                
            with torch.no_grad(), torch.cuda.amp.autocast():
                self.trainer_forward(container)
            outputs.append(
                container['model_output'].detach().cpu()
            )
            indexes.append(
                container['indexes'].detach().cpu()
            )
            torch.cuda.empty_cache()
            #if num_batch > 10:
            #   break
        return torch.cat(indexes), torch.cat(outputs)

    
    def run(self, train_loader, val_loader):
        best_metric = -1
        for epoch_num in range(self.epochs):
            self.run_epoch(train_loader, epoch_num)
            
            if (epoch_num + 1) % self.val_step == 0:
                val_metric = self.run_eval(val_loader, epoch_num)

                if self.use_wandb:
                    if best_metric != -1:
                        wandb.summary['val_score'] = max(
                            wandb.summary['val_score'], best_metric
                        )
                    else:
                        wandb.summary['val_score'] = best_metric
            
                if val_metric > best_metric:
                    best_metric = val_metric
                    self.save_checkpoint(
                        f'weights/{self.exp_name}_best.pt'
                    )
            self.save_checkpoint(
                f'weights/{self.exp_name}_last.pt'
            )


class DefaultTrainerFP16(DefaultTrainer):
    def __init__(self, **kwargs):
        super(DefaultTrainerFP16, self).__init__(**kwargs)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def trainer_forward(self, container):
        with torch.cuda.amp.autocast():
            self.model(container)

    def optimizer_step(self, container):
        total_loss = container['losses']['total_loss'] / self.accum_grad_steps
        self.scaler.scale(total_loss).backward()

        if self.train_step % self.accum_grad_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        self.train_step += 1