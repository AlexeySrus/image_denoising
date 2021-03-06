import torch
from denoising_pipeline.architectures.simple_denoising_net import \
    SimpleDenoisingNet
from denoising_pipeline.architectures.video_model import \
    VideoImprovingNet as SequentialNet
from denoising_pipeline.architectures.frequency_separation_model import \
    DenoisingNet as FrequencySeparationNet
from denoising_pipeline.architectures.MWCNNv2.mwcnnv2 import MWCNN
from denoising_pipeline.architectures.lambda_net import LambdaNet


def build_model_from_config(config: dict) -> torch.nn.Module:
    window_size = config['model']['window_size']

    if config['model']['architecture'] == 'simple':
        denoising_model = SimpleDenoisingNet()
    elif config['model']['architecture'] == 'sequential':
        denoising_model = SequentialNet(
            **config['model'][config['model']['architecture']]
        )
    elif config['model']['architecture'] == 'advanced':
        denoising_model = FrequencySeparationNet(
            shape=(window_size, window_size),
            n1=config['model'][
                config['model']['architecture']
            ]['separate_filters_count'],
            n2=config['model'][
                config['model']['architecture']
            ]['union_filters_count']
        )
    elif config['model']['architecture'] == 'wavelet':
        denoising_model = MWCNN(
            **config['model'][config['model']['architecture']]
        )
    elif config['model']['architecture'] == 'lambda':
        denoising_model = LambdaNet()
    else:
        raise NotImplementedError(
            'Set not supported model architecture: {}'.format(
                config['model']['architecture']
            )
        )

    return denoising_model
