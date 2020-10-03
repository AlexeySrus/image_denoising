# Image denoising realisation

### Setup instruction

#### Install requirements
Run follow command:
```shell script
pip3 install -r requirements.txt
```

#### Download dataset
By follow [link](https://yadi.sk/d/hVr5kLqNfMGILA "Yandex Disk") you can download dataset.
And extract this archive by follow path: `data/`

For downloading and extracting dataset you need ~30 Gb free space on your disk.

#### Prepare dataset to train loader
Run follow commands:
```shell script
python3 denoising_pipeline/scripts/preprocess_dataset.py \
  --dataset-path=data/dataset/after_iphone_denoising/ --verbose
```
```shell script
python3 denoising_pipeline/scripts/preprocess_dataset.py \
  --dataset-path=data/dataset/real_sense_noise/ --verbose
```
```shell script
python3 denoising_pipeline/scripts/preprocess_dataset.py \
  --dataset-path=data/dataset/webcam/ --verbose
```

### Training

#### Configure training pipeline 

Copy configuration file from `denoising_pipeline/configuration/example_train_config.yml`
to `data/` directory.

Or you can copy from follow YAML example:
```yaml
visualization:
  use_visdom: True
  visdom_port: 9000
  visdom_server: 'http://localhost'

  image:
    every: 10
    scale: 2

model:
  window_size: 224

  architecture: 'simple'  # choose from 'simple', 
                          # 'sequential' (work only with series dataset type), 
                          # 'advanced' (use image frequency decomposition)
                          # 'wavelet'

  sequential:
    series_size: 5
    n: 5
    residuals: True
    filters_per_image: 32

  advanced:
    separate_filters_count: 1
    union_filters_count: 1

  wavelet:
    n_features: 64
    activations: 'relu' # choose from 'relu', 
                        # 'mush' (requires a large amount of GPU memory)

dataset:
  type: 'pair'  # choose from 'pair', 'series', 'sequential', 
                # 'separate' (requires a large amount of RAM)

  pair:
    image1_path: 'path to first image from pair'
    image2_path: 'path to second image from pair'

  series:
    images_series_folder: 'path to folder with images folder from one series'

  sequential:
    images_series_folder: 'path to folder with images folders from series'

  separate:
    images_series_folder: 'path to noisy images'
    clear_images_path: 'path to clear images'

train:
  optimizer: 'sgd' # choose from 'adam', 'adamw', 'nadam', 'radam', 'sgd'
  lr: 0.0000001
  weight_decay: 0

  loss: 'mse' // choose from 'mse', 'l1', 'fourier_loss' (experimental loss)

  epochs: 150
  batch_size: 1
  number_of_processes: 12

  distribution_learning:
    enable: False
    devices: ['cuda:0', 'cuda:1']

  save:
    checkpoint_folder: 'path to saved weights and logs'
    every: 10

  load_model: False
  load_optimizer: False

```

Then you should setup configuration file according to your needs.

#### Start training
For training denoising network use follow script: `python3 denoising_pipeline/train/train.py`.
Before you need setup environment (you need open terminal in repository folder):
```shell script
export PYTHONPATH=./
```

Training script description:
```shell script
usage: train.py [-h] [--config CONFIG] [--no-cuda]

Image denoising train script

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Path to configuration yml file.
  --no-cuda        disables CUDA training
```

If you use **visdom** service start ot by follow command:
```shell script
visdom -port=9000
```

Note: Run shell scripts by **screen** service for convenience.

### Inference

Before you need setup environment (you need open terminal in repository folder):
```shell script
export PYTHONPATH=./
```

#### Image inference
For inference on image use follow script: `denoising_pipeline/test/inference_by_image.py`


Image inference script description:
```shell script
usage: inference_by_image.py [-h] [--config CONFIG] 
  \--model-weights MODEL_WEIGHTS 
  \--input-image INPUT_IMAGE 
  \--output-image OUTPUT_IMAGE [--no-cuda]

Video denoising train script

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration yml file.
  --model-weights MODEL_WEIGHTS
                        Path to model_estimator weights
  --input-image INPUT_IMAGE
                        Path to image file
  --output-image OUTPUT_IMAGE
                        Path to result image file
  --no-cuda             disables CUDA training
```

#### Video inference
For inference on video use follow script: `denoising_pipeline/test/inference_by_video.py`

Video inference script description:
```shell script
usage: inference_by_video.py [-h] [--config CONFIG] 
  \--model-weights MODEL_WEIGHTS 
  \--input-video INPUT_VIDEO 
  \--output-video OUTPUT_VIDEO [--no-cuda]

Video denoising train script

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration yml file.
  --model-weights MODEL_WEIGHTS
                        Path to model_estimator weights
  --input-video INPUT_VIDEO
                        Path to video file
  --output-video OUTPUT_VIDEO
                        Path to result video file
  --no-cuda             disables CUDA training
```

#### Pretrained models

Pretrained simaple and advanced models you can download by follow [link](https://yadi.sk/d/NA6Rg5S2JPpFdw "Yandex Disk").