import torch

from utils import get_tensor_size, identity


class Preprocessors:
    def __init__(self, args, models):
        self.model_names = list(models.keys())
        self.preprocessors = {k: Preprocessor(model, args) for k, model in models.items()}

    def __call__(self, row):
        return {model_name: self.run_model(model_name, row) for model_name in self.model_names}

    def run_model(self, model_name, row):
        return self.preprocessors[model_name](row)


class Preprocessor:
    def __init__(self, model, args):
        if args.computation.device == 'cuda' and hasattr(model, 'module'):
            _model = model.module
        else:
            _model = model
        if hasattr(_model, 'get_preprocessor'):
            self.preprocessor = _model.get_preprocessor()
        else:
            self.preprocessor = identity

    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)

    def preprocess(self, video):
        with torch.no_grad():
            video = _preprocess(self.preprocessor, video)
        return video


def _preprocess(preprocessor, data):
    if data is None or data[0][0] is None or data[1][0] is None:
        return None
    else:
        output = preprocessor(*data)  # {'data': preprocessed, 'fps': fps}
        if 'data' in output and get_tensor_size(output['data']) == 0:
            return None  # no element
        else:
            return output
