from collections import defaultdict


class StatKeeper():
    def __init__(self):
        self.losses = defaultdict(list)
    
    def step(self, container):
        for loss_name, loss_value in container['losses'].items():
            loss_value = float(loss_value)
            self.losses[loss_name].append(loss_value)
    
    def get_stat(self):
        ret_dict = {}
        for loss_name, losses in self.losses.items():
            ret_dict[loss_name + '_avg'] = sum(losses) / len(losses)
        return ret_dict