_loss_entrypoints = {}


def register_loss(fn):
    loss_name = fn.__name__
    _loss_entrypoints[loss_name] = fn
    return fn


def build_loss(config):
    loss_type = config.pop('type')
    loss_fn = _loss_entrypoints[loss_type]
    loss = loss_fn(**config)
    return loss