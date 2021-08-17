import torch
from torch import nn
from einops import rearrange

from tqdm import tqdm
from .utils_vggish import _preprocess


'''
VGGish model from https://github.com/harritaylor/torchvggish
input: either
    - wav file path
    - wav data converted to [-1.0, +1.0]
output: depends on the postprocess flag
    - True: 128D (NN)
    - False: 128D (NN, PCA transformeration, 8-bit quantization)
'''


def preprocess(visual, audio):
    data, fps = audio
    if torch.is_tensor(data):
        data = data.numpy()
    if data.shape[0] == 0:
        print('To short a video (< 1 min). Skipping the video.')
        preprocessed = None
    else:
        try:
            preprocessed = _preprocess(data, fps)
        except Exception as e:
            print('VGGISH preprocessing ERROR. Skipping the video.')
            print('fps', fps)
            print('data_shape', data.shape)
            print(e)
            preprocessed = None

    return {'data': preprocessed, 'fps': fps}


class Vggish(nn.Module):
    args = {
        'postprocess': False
    }
    output_dims = 128
    model_tag = {
        'name': 'VGGish',
        'dataset': 'YouTube-8M'
    }

    def __init__(self, args):
        super().__init__()

        torch.hub.set_dir(str(args.data.cache_dir))
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.model.postprocess = args.postprocess
        self.model.preprocess = False

    @classmethod
    def download(cls, args):
        torch.hub.set_dir(str(args.data.cache_dir))
        model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        return model

    def get_preprocessor(self):
        return preprocess

    def forward(self, data):
        B = data.shape[0]  # BNCHW
        data = rearrange(data, 'b n c h w -> (b n) c h w')
        data = self.model.forward(data)
        data = rearrange(data, '(b n) c -> b n c', b=B)
        data = data.mean(dim=1)  # B 128
        return data


class LayerVggish(Vggish):
    args = {
        'num_layers': 5,
        'postprocess': False
    }
    output_dims = [64, 128, 256, 512, 128]

    def __init__(self, args):
        super().__init__(args)
        self.num_layers = args.num_layers

    def forward(self, data):
        B = data.shape[0]  # BNCHW
        data = rearrange(data, 'b n c h w -> (b n) c h w')
        res = self.model_forward(data)
        res_pooled = []
        for data in res:
            data = rearrange(data, '(b n) c -> b n c', b=B)
            data = data.mean(dim=1)  # B 128
            res_pooled.append(data)
        return res_pooled

    def model_forward(self, x, fs=None):
        model = self.model
        if model.preprocess:
            x = model._preprocess(x, fs)
        x, res = self.vgg_forward(x)
        if model.postprocess:
            x = model._postprocess(x)
            res.append(x.detach().cpu())
        return res

    def vgg_forward(self, inputs):
        model = self.model
        res = self.run_features(inputs)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        res_resized = []
        for feat in res:
            feat = feat.mean(-1).mean(-1)  # avgpool image size in mel features
            res_resized.append(feat.detach().cpu())

        x = res[-1]
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = model.embeddings(x)
        del res
        res_resized.append(x.detach().cpu())
        return x, res_resized

    def run_features(self, x):
        features = self.model.features

        # segment blocks by MaxPool2d
        blocks = [i + 1 for i, m in enumerate(features) if isinstance(m, torch.nn.modules.pooling.MaxPool2d)]
        blocks = zip([0] + blocks[:-1], blocks)
        blocks = [list(range(x[0], x[1])) for x in blocks]
        blocks = [nn.Sequential(*[features[i] for i in block]) for block in blocks]
        res = []
        for block in blocks:
            x = block(x)
            res.append(x)
        return res
