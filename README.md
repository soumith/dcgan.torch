

#Imagenet dataset
## Preprocessing


## Training
DATA_ROOT=[PATH_TO_IMAGENET]/train dataset=folder th main.lua

#CelebA dataset

## Preprocessing
mkdir celebA
cd celebA
# download img_align_celeba.zip from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html "Align&Cropped Images"
unzip img_align_celeba.zip
cd ..
DATA_ROOT=celebA th data/crop_celebA.lua

## Training
DATA_ROOT=celebA dataset=folder th main.lua