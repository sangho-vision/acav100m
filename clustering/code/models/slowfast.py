from torch import nn


'''
model_tags = {
    'Kinetics/c2/SLOWFAST_8x8_R50':
        {
            'name': 'SLOWFAST_8x8_R50',
            'dataset': 'kinetics-400',
        }
}
'''


class SlowFast(nn.Module):
    args = {
        'slowfast_config': 'Kinetics/c2/SLOWFAST_8x8_R50'
    }
    model_tag = {
        'name': 'SLOWFAST_8x8_R50',
        'dataset': 'kinetics-400',
    }
    output_dims = 2304


class LayerSlowFast(SlowFast):
    args = {
        'slowfast_config': 'Kinetics/c2/SLOWFAST_8x8_R50',
        'num_layers': 5
    }
    output_dims = [88, 352, 704, 1408, 2304]
