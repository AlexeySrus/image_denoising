from denoising_pipeline.architectures.MWCNNv2 import common
import torch
import torch.nn as nn

from denoising_pipeline.utils.activations import Mish


class MWCNN(nn.Module):
    def __init__(self,
                 n_features: int = 64,
                 n_colors: int = 3,
                 conv: nn.Module = common.default_conv,
                 activations: str = 'relu'):
        super(MWCNN, self).__init__()
        n_feats = n_features
        kernel_size = 3
        self.scale_idx = 0
        nColor = n_colors
        bias = False
        batch_norm = False

        if activations == 'relu':
            act = nn.ReLU(True)
        elif activations == 'mish':
            act = Mish()
        else:
            raise RuntimeError(
                'Get unsupported activation function: {}'.format(activations)
            )

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        m_head = [common.BBlock(conv, nColor, n_feats, kernel_size, act=act, bias=bias, bn=batch_norm)]
        d_l0 = []
        d_l0.append(common.DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bias=bias, bn=batch_norm))


        d_l1 = [common.BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bias=bias, bn=batch_norm)]
        d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bias=bias, bn=batch_norm))

        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bias=bias, bn=batch_norm))
        d_l2.append(common.DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bias=bias, bn=batch_norm))
        pro_l3 = []
        pro_l3.append(common.BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bias=bias, bn=batch_norm))
        pro_l3.append(common.DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bias=bias, bn=batch_norm))
        pro_l3.append(common.DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bias=bias, bn=batch_norm))
        pro_l3.append(common.BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bias=bias, bn=batch_norm))

        i_l2 = [common.DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bias=bias, bn=batch_norm)]
        i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bias=bias, bn=batch_norm))

        i_l1 = [common.DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bias=bias, bn=batch_norm)]
        i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bias=bias, bn=batch_norm))

        i_l0 = [common.DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bias=bias, bn=batch_norm)]

        m_tail = [conv(n_feats, nColor, kernel_size)]

        self.pad_size = (kernel_size // 2) + 1 - 1
        for layer_seq in [m_head, d_l0, d_l2, pro_l3, i_l2, i_l1, i_l0]:
            for layer in layer_seq:
                self.pad_size += layer.pad_size

        # print('Model padding size: {}'.format(self.pad_size))

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        x = self.tail(self.i_l0(x_)) + x

        return x

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[
            :,
            :,
            self.pad_size:-self.pad_size,
            self.pad_size:-self.pad_size
        ]

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
