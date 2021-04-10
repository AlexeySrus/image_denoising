import torch
from torch.nn import functional
from lambda_networks import LambdaLayer
from denoising_pipeline.utils.activations import mish


class LambdaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = LambdaLayer(
            dim=3,  # channels going in
            dim_out=16,  # channels out
            r=23,         # the receptive field for relative positional encoding (23 x 23)
            dim_k=32,  # key dimension
            heads=1,  # number of heads, for multi-query
            dim_u=4  # 'intra-depth' dimension
        )
        self.layer2 = LambdaLayer(
            dim=16,  # channels going in
            dim_out=3,  # channels out
            r=15,
            # the receptive field for relative positional encoding (23 x 23)
            dim_k=16,  # key dimension
            heads=1,  # number of heads, for multi-query
            dim_u=4  # 'intra-depth' dimension
        )

        self.last_conv = torch.nn.Conv2d(3, 3, 1, bias=False)

    def forward(self, x):
        y = self.layer1(x)
        y = torch.nn.functional.relu(y)
        y = self.layer2(y)
        return y
        # y = torch.nn.functional.relu(y)
        # return self.last_conv(y)
#
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class LambdaConv(nn.Module):
#     def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=23):
#         super(LambdaConv, self).__init__()
#         self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
#         self.local_context = True if m > 0 else False
#         self.padding = (m - 1) // 2
#
#         self.queries = nn.Sequential(
#             nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
#             nn.BatchNorm2d(k * heads)
#         )
#         self.keys = nn.Sequential(
#             nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
#         )
#         self.values = nn.Sequential(
#             nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
#             nn.BatchNorm2d(self.vv * u)
#         )
#
#         self.softmax = nn.Softmax(dim=-1)
#
#         if self.local_context:
#             self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)
#         else:
#             self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)
#
#     def forward(self, x):
#         n_batch, C, w, h = x.size()
#
#         queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k // heads, w * h
#         softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
#         values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h
#
#         lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values)
#         y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)
#
#         if self.local_context:
#             values = values.view(n_batch, self.uu, -1, w, h)
#             lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
#             lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)
#             y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
#         else:
#             lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
#             y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
#
#         out = y_c + y_p
#         out = out.contiguous().view(n_batch, -1, w, h)
#
#         return out
#
#
# class LambdaBottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(LambdaBottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#
#         self.conv2 = nn.ModuleList([LambdaConv(planes, planes)])
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.conv2.append(nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))
#         self.conv2.append(nn.BatchNorm2d(planes))
#         self.conv2.append(nn.ReLU())
#         self.conv2 = nn.Sequential(*self.conv2)
#
#         self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.conv2(out)
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# # reference
# # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=1000):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         # ImageNet 350 epochs training setup
#         # self.maxpool = nn.Sequential(
#         #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU()
#         # )
#
#         self.layer1 = self._make_layer(block, 64, num_blocks[0])
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
#             nn.Linear(512 * block.expansion, num_classes)
#         )
#
#     def _make_layer(self, block, planes, num_blocks, stride=1):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for idx, stride in enumerate(strides):
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.maxpool(out)
#
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#
#         out = self.avgpool(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out
#
#
# def LambdaResNet18():
#     return ResNet(LambdaBottleneck, [2, 2, 2, 2])
#
#
# def LambdaResNet50():
#     return ResNet(LambdaBottleneck, [3, 4, 6, 3])
#
#
# def LambdaResNet152():
#     return ResNet(LambdaBottleneck, [3, 8, 36, 3])
#
#
# def LambdaResNet200():
#     return ResNet(LambdaBottleneck, [3, 24, 36, 3])
#
#
# def LambdaResNet270():
#     return ResNet(LambdaBottleneck, [4, 29, 53, 4])
#
#
# def LambdaResNet350():
#     return ResNet(LambdaBottleneck, [4, 36, 72, 4])
#
#
# def LambdaResNet420():
#     return ResNet(LambdaBottleneck, [4, 44, 87, 4])