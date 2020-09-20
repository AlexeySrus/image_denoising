from argparse import ArgumentParser
import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = ArgumentParser(
        description='Estimate PSNR measure between two images'
    )
    parser.add_argument(
        '--clean-image', type=str, required=True
    )
    parser.add_argument(
        '--noise-image', type=str, required=True
    )

    return parser.parse_args()


def main():
    args = parse_args()

    clear = np.array(Image.open(args.clean_image).convert('RGB'))
    noise = np.array(Image.open(args.noise_image).convert('RGB'))

    psnr = cv2.PSNR(clear, noise)
    print('PSNR: {}'.format(psnr))


if __name__ == '__main__':
    main()
