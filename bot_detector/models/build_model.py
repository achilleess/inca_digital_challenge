_model_entrypoints = {}


def register_model(fn):
    model_name = fn.__name__
    _model_entrypoints[model_name] = fn
    return fn


def build_model(config):
    model_type = config.pop('type')
    model_fn = _model_entrypoints[model_type]
    model = model_fn(**config)
    return model