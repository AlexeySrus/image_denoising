import torch


class Discriminator(torch.nn.Module):
    def __init__(self, input_shape: tuple = (3, 224, 224)):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(torch.nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(torch.nn.BatchNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            layers.append(torch.nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(torch.nn.BatchNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(
            torch.nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1)
        )

        self.discriminator_model = torch.nn.Sequential(*layers)

    def forward(self, img):
        return self.discriminator_model(img)
