import torch
import torchvision
import tqdm
import os
import re
from denoising_pipeline.utils.losses import l2
from denoising_pipeline.utils.losses import acc as acc_function
from denoising_pipeline.utils.tensor_utils import center_pad_tensor_like
from denoising_pipeline.model_estimator.model_inference import denoise_inference


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Model:
    def __init__(self, net, _device='cpu', callbacks_list=None,
                 distributed_learning=False, distributed_devices=None,
                 discriminator=None, feature_extractor=None):
        self.device = torch.device('cpu' if _device == 'cpu' else 'cuda')

        self.model = net.to(self.device)
        self.discriminator = None \
            if discriminator is None else \
            discriminator.to(self.device)
        self.feature_extractor = None
        if feature_extractor is None and discriminator is not None:
            self.feature_extractor = torch.nn.Sequential(
                *list(
                    torchvision.models.vgg19(
                        pretrained=True
                    ).features.children()
                )[:35]
            ).to(self.device)
        self.callbacks = [] if callbacks_list is None else callbacks_list
        self.last_n = 0
        self.last_optimiser_state = None
        self.last_generator_optimiser_state = None
        self.last_discriminator_optimiser_state = None
        self.distributed_learning = distributed_learning
        self.distributed_devices = distributed_devices

        self.distributed_learning = self.distributed_learning and torch.cuda.device_count() > 1 and self.device != 'cpu'
        self.distributed_learning = self.distributed_learning and (
            len(
                distributed_devices) > 1 if distributed_devices is not None else True)
        self.distributed_devices = distributed_devices

        if distributed_devices is not None and _device != 'cpu' and distributed_learning:
            self.device = torch.device(distributed_devices[0])

        if self.distributed_learning:
            print('Target devices: {}'.format(self.distributed_devices))

            devices_inds = None if self.distributed_devices is None \
                else [int(d.split(':')[-1]) for d in
                      self.distributed_devices]
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=devices_inds)

            if self.discriminator is not None:
                self.discriminator = torch.nn.DataParallel(
                    self.discriminator,
                    device_ids=devices_inds
                )
                self.feature_extractor = torch.nn.DataParallel(
                    self.feature_extractor,
                    device_ids=devices_inds
                )

    def fit(self,
            train_loader,
            optimizer,
            epochs=1,
            loss_function=l2,
            validation_loader=None,
            verbose=False,
            init_start_epoch=1,
            acc_f=acc_function,
            is_epoch_scheduler=True):
        """
        Model train method
        Args:
            train_loader: DataLoader
            optimizer: optimizer from torch.optim with initialized parameters
            or tuple of (optimizer, scheduler)
            epochs: epochs count
            loss_function: Loss function
            validation_loader: DataLoader
            verbose: print evaluate validation prediction
            init_start_epoch: start epochs number
        Returns:
        """
        scheduler = None
        if type(optimizer) is tuple:
            scheduler = optimizer[1]
            optimizer = optimizer[0]

        for epoch in range(init_start_epoch, epochs + 1):
            self.model.train()

            batches_count = len(train_loader)
            avg_epoch_loss = 0
            avg_epoch_acc = 0

            if scheduler is not None and is_epoch_scheduler:
                scheduler.step(epoch)

            self.last_n = epoch

            test_loss_ = 0

            with tqdm.tqdm(total=batches_count) as pbar:
                for i, batch in enumerate(train_loader):
                    self.last_optimiser_state = optimizer.state_dict()

                    x = tuple([
                        batch[i].to(self.device)
                        for i in range(0, len(batch) - 1)
                    ])
                    y_true = batch[-1].to(self.device)

                    optimizer.zero_grad()
                    y_pred = self.model(*x)

                    loss = loss_function(
                        y_pred,
                        center_pad_tensor_like(y_true, y_pred)
                    )

                    if not self.distributed_learning:
                        loss.backward()
                    else:
                        loss.mean().to(
                            device='cuda:0', dtype=torch.float).backward()

                    optimizer.step()

                    acc = 0
                    # acc = acc_f(
                    #     flatten(y_pred),
                    #     flatten(crop_batch_by_center(y_true, y_pred.shape))
                    # )

                    pbar.postfix = \
                        'Epoch: {}/{}, loss: {:.8f}, acc: {:.8f}, lr: {:.8f}'.format(
                            epoch,
                            epochs,
                            loss.item() / train_loader.batch_size,
                            acc,
                            get_lr(optimizer)
                        )
                    avg_epoch_loss += \
                        loss.item() / train_loader.batch_size / batches_count

                    avg_epoch_acc += acc

                    for cb in self.callbacks:
                        cb.per_batch({
                            'model': self,
                            'loss': loss.item() / train_loader.batch_size,
                            'n': (epoch - 1) * batches_count + i + 1,
                            'x': center_pad_tensor_like(x[0], y_pred),
                            'y_pred': y_pred,
                            'y_true': center_pad_tensor_like(y_true, y_pred),
                            'acc': acc,
                            'loss_xn': loss_function(
                                y_pred,
                                center_pad_tensor_like(x[0], y_pred)
                            )
                        })

                    test_loss_ += \
                        (
                                loss.item() +
                                loss_function(
                                    y_pred,
                                    center_pad_tensor_like(x[0], y_pred)
                                ).item()
                        ) / train_loader.batch_size / batches_count

                    pbar.update(1)

            test_loss = None
            test_acc = None

            if validation_loader is not None:
                test_loss, test_acc = self.evaluate(
                    validation_loader, loss_function, verbose
                )
                self.model.train()

            for cb in self.callbacks:
                cb.per_epoch({
                    'model': self,
                    'loss': avg_epoch_loss,
                    'val loss': test_loss_,
                    'n': epoch,
                    'optimize_state': optimizer.state_dict(),
                    'acc': avg_epoch_acc,
                    'val acc': test_acc
                })

            if scheduler is not None and not is_epoch_scheduler:
                scheduler.step(avg_epoch_loss)

    def adversarial_fit(self,
                        train_loader,
                        generator_optimizer,
                        discriminator_optimizer,
                        discriminator_output_shape,
                        epochs=1,
                        pixel_wise_loss_function=l2,
                        loss_weights: tuple = (1.0, 1.0, 1.0),
                        validation_loader=None,
                        verbose=False,
                        init_start_epoch=1,
                        acc_f=acc_function):
        """
        Model train method
        Args:
            train_loader: DataLoader
            generator_optimizer: optimizer from torch.optim with initialized
                                    generator parameters
            discriminator_optimizer: optimizer from torch.optim with initialized
                                        discriminator parameters
            discriminator_output_shape:
            epochs: epochs count
            loss_weights: tuple of
                (content loss scale, adversarial scale, pixel-wise loss scale)
            pixel_wise_loss_function: Loss function
            validation_loader: DataLoader
            verbose: print evaluate validation prediction
            init_start_epoch: start epochs number
        Returns:
        """
        assert self.discriminator is not None

        for epoch in range(init_start_epoch, epochs + 1):
            self.model.train()

            batches_count = len(train_loader)
            avg_epoch_generator_loss = 0
            avg_epoch_discriminator_loss = 0
            avg_epoch_acc = 0
            self.last_n = epoch

            test_loss_ = 0

            with tqdm.tqdm(total=batches_count) as pbar:
                for i, batch in enumerate(train_loader):
                    self.last_generator_optimiser_state = \
                        generator_optimizer.state_dict()
                    self.last_discriminator_optimiser_state = \
                        discriminator_optimizer.state_dict()

                    x = tuple([
                        torch.autograd.Variable(batch[i]).to(self.device)
                        for i in range(0, len(batch) - 1)
                    ])
                    y_true = torch.autograd.Variable(batch[-1].to(self.device))

                    valid = torch.autograd.Variable(
                        torch.ones(x[0].size(0), *discriminator_output_shape),
                        requires_grad=False
                    ).to(self.device)
                    fake = torch.autograd.Variable(
                        torch.zeros(x[0].size(0), *discriminator_output_shape),
                        requires_grad=False
                    ).to(self.device)

                    # ------------------
                    #  Train Generator
                    # ------------------
                    generator_optimizer.zero_grad()
                    y_pred = self.model(*x)

                    pixel_wise_loss = pixel_wise_loss_function(
                        y_pred,
                        center_pad_tensor_like(y_true, y_pred)
                    )

                    used_only_pixel_wise_loss = False
                    if torch.rand(1).numpy()[0] - 0.9 > -1E-5:
                        used_only_pixel_wise_loss = True

                    pred_real = self.discriminator(y_true).detach()
                    pred_fake = self.discriminator(y_pred)

                    # Adversarial loss (relativistic average GAN)
                    loss_adversarial = \
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            pred_fake - pred_real.mean(0, keepdim=True),
                            valid
                        )

                    # Content loss
                    gen_features = self.feature_extractor(y_pred)
                    real_features = self.feature_extractor(y_true).detach()
                    loss_content = torch.nn.functional.l1_loss(
                        gen_features, real_features
                    )

                    # Total generator loss
                    total_generator_loss = loss_weights[0] * loss_content +\
                                           loss_weights[1] * loss_adversarial +\
                                           loss_weights[2] * pixel_wise_loss

                    if not self.distributed_learning:
                        if used_only_pixel_wise_loss:
                            pixel_wise_loss.backward()
                        else:
                            total_generator_loss.backward()
                    else:
                        if used_only_pixel_wise_loss:
                            pixel_wise_loss.mean().to(
                                device='cuda:0', dtype=torch.float).backward()
                        else:
                            total_generator_loss.mean().to(
                                device='cuda:0', dtype=torch.float).backward()

                    generator_optimizer.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    discriminator_optimizer.zero_grad()

                    pred_real = self.discriminator(y_true)
                    pred_fake = self.discriminator(y_pred.detach())

                    # Adversarial loss for real and fake images
                    # (relativistic average GAN)
                    loss_real = \
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            pred_real - pred_fake.mean(0, keepdim=True),
                            valid
                        )
                    loss_fake = \
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            pred_fake - pred_real.mean(0, keepdim=True),
                            fake
                        )

                    # Total loss
                    total_discriminator_loss = (loss_real + loss_fake) / 2

                    if not used_only_pixel_wise_loss:
                        if not self.distributed_learning:
                            total_discriminator_loss.backward()
                        else:
                            total_discriminator_loss.mean().to(
                                device='cuda:0', dtype=torch.float).backward()

                        discriminator_optimizer.step()

                    acc = 0
                    # acc = acc_f(
                    #     flatten(y_pred),
                    #     flatten(crop_batch_by_center(y_true, y_pred.shape))
                    # )

                    pbar.postfix = \
                        'Epoch: {}/{}, generator_loss: {:.8f}, ' \
                        'discriminator_loss: {:.8f}'.format(
                            epoch,
                            epochs,
                            total_generator_loss.item() /
                            train_loader.batch_size,
                            total_discriminator_loss.item() /
                            train_loader.batch_size
                        )
                    avg_epoch_generator_loss += \
                        total_generator_loss.item() / \
                        train_loader.batch_size / batches_count
                    avg_epoch_discriminator_loss += \
                        total_discriminator_loss.item() / \
                        train_loader.batch_size / batches_count

                    avg_epoch_acc += acc

                    for cb in self.callbacks:
                        cb.per_batch({
                            'model': self,
                            'generator_loss':
                                total_generator_loss.item() /
                                train_loader.batch_size,
                            'generator_content_loss':
                                loss_content.item() /
                                train_loader.batch_size,
                            'generator_pixel_wise_loss':
                                pixel_wise_loss.item() /
                                train_loader.batch_size,
                            'generator_adversarial_loss':
                                loss_adversarial.item() /
                                train_loader.batch_size,
                            'discriminator_loss':
                                total_discriminator_loss.item() /
                                train_loader.batch_size,
                            'n': (epoch - 1) * batches_count + i + 1,
                            'x': center_pad_tensor_like(x[0], y_pred),
                            'y_pred': y_pred,
                            'y_true': center_pad_tensor_like(y_true, y_pred),
                            'acc': acc,
                            'loss_xn': pixel_wise_loss_function(
                                y_pred,
                                center_pad_tensor_like(x[0], y_pred)
                            )
                        })

                    test_loss_ += \
                        (
                            pixel_wise_loss_function(
                                y_pred,
                                center_pad_tensor_like(x[0], y_pred)
                            ).item()
                        ) / train_loader.batch_size / batches_count

                    pbar.update(1)

            test_loss = None
            test_acc = None

            if validation_loader is not None:
                test_loss, test_acc = self.evaluate(
                    validation_loader, pixel_wise_loss_function, verbose
                )
                self.model.train()

            for cb in self.callbacks:
                cb.per_epoch({
                    'model': self,
                    'generator_loss': avg_epoch_generator_loss,
                    'discriminator_loss': avg_epoch_discriminator_loss,
                    'val loss': test_loss_,
                    'n': epoch,
                    'generator_optimize_state':
                        generator_optimizer.state_dict(),
                    'discriminator_optimize_state':
                        discriminator_optimizer.state_dict(),
                    'acc': avg_epoch_acc,
                    'val acc': test_acc
                })

    def evaluate(self,
                 test_loader,
                 loss_function=l2,
                 verbose=False,
                 acc_f=acc_function):
        """
        Test model
        Args:
            test_loader: DataLoader
            loss_function: loss function
            verbose: print progress

        Returns:

        """
        self.model.eval()

        test_loss = 0
        test_acc = 0

        with torch.no_grad():
            set_range = tqdm.tqdm(test_loader) if verbose else test_loader
            for _x, _y_true in set_range:
                x = _x.to(self.device)
                y_true = _y_true.to(self.device)
                y_pred = self.model(x)
                test_loss += loss_function(
                    y_pred, y_true
                ).item() / test_loader.batch_size / len(test_loader)
                test_acc += \
                    acc_f(y_pred, y_true).detach().numpy() / len(test_loader)

        return test_loss, test_acc

    def predict(self,
                image,
                window_size=224,
                batch_size=1,
                verbose=False):
        """
        Predict by cv2 frame (numpy uint8 array)
        Args:
            image: image in numpy uint8 RGB format
            window_size: window size
            batch_size: inference batch size
            verbose: print prediction progress

        Returns:

        """

        with torch.no_grad():
            return denoise_inference(
                image=image,
                model=self.model,
                window_size=window_size,
                batch_size=batch_size,
                device=self.device,
                verbose=verbose
            )

    def set_callbacks(self, callbacks_list):
        self.callbacks = callbacks_list

    def get_parameters_count(self):
        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

    def save(self, path):
        self.model = self.model.to('cpu')
        torch.save(
            self.model.state_dict()
            if not self.distributed_learning else
            self.model.module.state_dict(),
            path
        )
        self.model = self.model.to(self.device)

    def load(self, path):
        if not self.distributed_learning:
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            self.model.module.load_state_dict(
                torch.load(path, map_location='cpu')
            )
        self.model.eval()
        self.model = self.model.to(self.device)

    def __del__(self):
        for cb in self.callbacks:
            if self.discriminator is not None:
                cb.early_stopping(
                    {
                        'model': self,
                        'n': self.last_n,
                        'generator_optimize_state':
                            self.last_generator_optimiser_state,
                        'discriminator_optimize_state':
                            self.last_discriminator_optimiser_state
                    }
                )
            else:
                cb.early_stopping(
                    {
                        'model': self,
                        'n': self.last_n,
                        'optimize_state':
                            self.last_optimiser_state
                    }
                )


def get_last_epoch_weights_path(checkpoints_dir, log=None):
    """
    Get last epochs weights from target folder
    Args:
        checkpoints_dir: target folder
        log: logging, default standard print
    Returns:
        (
            path to current weights file,
            path to current optimiser file,
            current epoch number
        )
    """
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        return None, None, 0

    weights_files_list = [
        matching_f.group()
        for matching_f in map(
            lambda x: re.match('model-\d+.trh', x),
            os.listdir(checkpoints_dir)
        ) if matching_f if not None
    ]

    if len(weights_files_list) == 0:
        return None, None, 0

    weights_files_list.sort(key=lambda x: -int(x.split('-')[1].split('.')[0]))

    if log is not None:
        log('LOAD MODEL PATH: {}'.format(
            os.path.join(checkpoints_dir, weights_files_list[0])
        ))

    n = int(
        weights_files_list[0].split('-')[1].split('.')[0]
    )

    return os.path.join(checkpoints_dir,
                        weights_files_list[0]
                        ), \
           os.path.join(checkpoints_dir, 'optimize_state-{}.trh'.format(n)), n
