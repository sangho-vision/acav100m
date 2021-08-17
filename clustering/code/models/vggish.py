from torch import nn


class Vggish(nn.Module):
    args = {
        'postprocess': False
    }
    output_dims = 128
    model_tag = {
        'name': 'VGGish',
        'dataset': 'YouTube-8M'
    }


class LayerVggish(Vggish):
    args = {
        'num_layers': 5,
        'postprocess': False
    }
    output_dims = [64, 128, 256, 512, 128]
