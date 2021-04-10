import torch
import argparse
import os
from shutil import copyfile
import yaml
import numpy as np
import torch.nn.functional as F
from denoising_pipeline.model_estimator.model import \
    Model, get_last_epoch_weights_path
from denoising_pipeline.utils.callbacks import (SaveModelPerEpoch, VisPlot,
                                               SaveOptimizerPerEpoch,
                                               SaveGANOptimizerPerEpoch,
                                               VisImageForAE)
from denoising_pipeline.datasets.configure_dataset import \
    build_dataset_from_config
from torch.utils.data import DataLoader
from denoising_pipeline.utils.optimizers import Nadam, RangerAdam as Radam
from denoising_pipeline.utils.losses import FourierImagesLoss
from denoising_pipeline.architectures.configure_model import \
    build_model_from_config
from denoising_pipeline.architectures.discriminator import Discriminator


def parse_args():
    parser = argparse.ArgumentParser(description='Image denoising train script')
    parser.add_argument('--config', required=False, type=str,
                        default='../configuration/example_train_config.yml',
                        help='Path to configuration yml file.'
                        )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert (config['model']['architecture'] != 'sequential') or \
           (
                   config['model']['architecture'] == 'sequential' and
                   config['dataset']['type'] == 'sequential'
           ) or \
           (
                   config['dataset']['type'] == 'pair' and
                   config['model']['architecture'] == 'sequential' and
                   config['model']['sequential']['series_size'] == 1
           )

    batch_size = config['train']['batch_size']
    n_jobs = config['train']['number_of_processes']
    epochs = config['train']['epochs']
    window_size = config['model']['window_size']
    is_adversarial_train = config['train']['training_approach'] == 'adversarial'

    checkpoint_folder = os.path.join(
        os.path.dirname(__file__),
        config['train']['save']['checkpoint_folder']
    )

    if not os.path.isdir(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    copyfile(
        args.config,
        os.path.join(
            checkpoint_folder,
            os.path.basename(args.config)
        )
    )

    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'nadam': Nadam,
        'radam': Radam,
        'sgd': torch.optim.SGD,
    }

    losses = {
        'mse': F.mse_loss,
        'l1': F.l1_loss,
        'fourier_loss': FourierImagesLoss(
            loss_sum_coeffs=(1, 0.5),
            image_shape=(window_size, window_size),
            four_normalized=True
        )
    }

    denoising_model = build_model_from_config(config)

    discriminator = None
    if is_adversarial_train:
        discriminator = Discriminator(
            input_shape=(3, window_size, window_size)
        )

    model = Model(
        denoising_model,
        device,
        distributed_learning=config['train']['distribution_learning']['enable'],
        distributed_devices=config['train']['distribution_learning']['devices'],
        discriminator=discriminator
    )

    callbacks = []

    callbacks.append(SaveModelPerEpoch(
        checkpoint_folder,
        config['train']['save']['every']
    ))

    if not is_adversarial_train:
        callbacks.append(SaveOptimizerPerEpoch(
            checkpoint_folder,
            config['train']['save']['every']
        ))
    else:
        callbacks.append(SaveGANOptimizerPerEpoch(
            checkpoint_folder,
            config['train']['save']['every']
        ))

    if config['visualization']['use_visdom']:
        plots = VisPlot(
            'Image denoising {} train'.format(
                config['train']['training_approach']
            ),
            server=config['visualization']['visdom_server'],
            port=config['visualization']['visdom_port']
        )

        if not is_adversarial_train:
            # plots.register_scatterplot('train loss per_batch', 'Batch number',
            #                            'Loss',
            #                            [
            #                                '{} between '
            #                                'predicted and ground truth'
            #                                ''.format(config['train']['loss']),
            #                                '{} between predicted and input'
            #                                ''.format(config['train']['loss'])
            #                            ])

            plots.register_scatterplot('train validation loss per_epoch',
                                       'Batch number',
                                       'Loss',
                                       [
                                           '{} train loss'.format(
                                               config['train']['loss']
                                           ),
                                           'double {} train loss'.format(
                                               config['train']['loss']
                                           )
                                       ])
        else:
            plots.register_scatterplot('train loss generator per_batch',
                                       'Batch number',
                                       'Loss',
                                       [
                                           'Generator loss',
                                       ])

            plots.register_scatterplot('train loss generator '
                                       'separating content per_batch',
                                       'Batch number',
                                       'Loss',
                                       [
                                           'Generator content loss',
                                       ])

            plots.register_scatterplot('train loss generator '
                                       'separating pixel per_batch',
                                       'Batch number',
                                       'Loss',
                                       [
                                           'Generator pixel-wise loss',
                                       ])

            plots.register_scatterplot('train loss generator '
                                       'separating adversarial per_batch',
                                       'Batch number',
                                       'Loss',
                                       [
                                           'Generator adversarial loss',
                                       ])

            # plots.register_scatterplot('train loss generator separating '
            #                            'per_batch',
            #                            'Batch number',
            #                            'Loss',
            #                            [
            #                                'Content loss',
            #                                'Pixel-wise loss',
            #                                'Adversarial loss',
            #                            ])

            plots.register_scatterplot('train loss discriminator per_batch',
                                       'Batch number',
                                       'Loss',
                                       [
                                           'Discriminator loss',
                                       ])

        callbacks.append(plots)

        callbacks.append(
            VisImageForAE(
                'Image visualisation',
                config['visualization']['visdom_server'],
                config['visualization']['visdom_port'],
                config['visualization']['image']['every'],
                scale=config['visualization']['image']['scale']
            )
        )

    model.set_callbacks(callbacks)

    series_size = 1 \
        if config['model']['architecture'] != 'sequential' else \
        config['model'][config['model']['architecture']]['series_size']

    dataset_loader = build_dataset_from_config(config)

    start_epoch = 0
    if config['train']['optimizer'] != 'sgd':
        optimizer = optimizers[config['train']['optimizer']](
            model.model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'],
            momentum=0.9,
            nesterov=True
        )

    discriminator_optimizer = None
    if is_adversarial_train:
        if config['train']['optimizer'] != 'sgd':
            discriminator_optimizer = optimizers[config['train']['optimizer']](
                model.discriminator.parameters(),
                lr=config['train']['lr'],
                weight_decay=config['train']['weight_decay']
            )
        else:
            discriminator_optimizer = torch.optim.SGD(
                model.discriminator.parameters(),
                lr=config['train']['lr'],
                weight_decay=config['train']['weight_decay'],
                momentum=0.9,
                nesterov=True
            )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        verbose=True
    )

    if config['train']['load_model'] or config['train']['load_optimizer']:
        weight_path, optim_path, start_epoch = get_last_epoch_weights_path(
            checkpoint_folder,
            print
        )

        if weight_path is not None:
            if config['train']['load_model']:
                model.load(weight_path)

            if config['train']['load_optimizer']:
                optimizer.load_state_dict(torch.load(optim_path,
                                                     map_location='cpu'))

    train_data = DataLoader(
        dataset_loader,
        batch_size=batch_size,
        num_workers=n_jobs,
        drop_last=True
    )

    print(
        'Count of model trainable parameters: {}'.format(
            model.get_parameters_count()
        )
    )

    print('Train model out shape: {}'.format(
        model.model(*tuple([
                               torch.FloatTensor(
                                   np.zeros((1, 3, window_size, window_size))
                               ).to(device)
                           ] * series_size)).shape
    ))

    if not is_adversarial_train:
        model.fit(
            train_data,
            (optimizer, scheduler),
            epochs,
            losses[config['train']['loss']],
            init_start_epoch=start_epoch + 1,
            validation_loader=None,
            is_epoch_scheduler=False
        )
    else:
        model.adversarial_fit(
            train_data,
            optimizer,
            discriminator_optimizer,
            epochs=epochs,
            discriminator_output_shape=discriminator.output_shape,
            pixel_wise_loss_function=losses[config['train']['loss']],
            loss_weights=(
                             config['train']['content_loss_scale'],
                             config['train']['adversarial_loss_scale'],
                             config['train']['pixel_wise_loss_scale']
                         ),
            init_start_epoch=start_epoch + 1,
            validation_loader=None
        )


if __name__ == '__main__':
    main()
