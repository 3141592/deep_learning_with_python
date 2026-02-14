#!/bin/bash

pip install -U gdown
mkdir -p ~/src/data/celeba_gan
cd ~/src/data/celeba_gan
gdown --id 0B7EVK8r0v71pZjFTYXZWM3FlRnM -O img_align_celeba.zip

unzip -q img_align_celeba.zip
