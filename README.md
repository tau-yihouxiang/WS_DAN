# Weakly Supervised Data Augmentation Network

This is the official TensorFlow implementation of WS-DAN.

[See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification](https://arxiv.org/abs/1901.09891).

## Compatibility
* The code is tested using Tensorflow r1.x under Ubuntu 16.04 with Python 2.x and Python 3.x.

* Recommend Environment: Anaconda

## Requirements
```
$ git clone git@github.com:tau-yihouxiang/WS_DAN.git
$ cd WS_DAN
$ python setup.py install
```

* opencv, tqdm
```
$ conda install -c menpo opencv
$ pip install tqdm
```

## Datasets and Pre-trained models
|   Datasets    | #Attention | Pre-trained model |
|:-------------:|:----------:|:------------------:|
|  [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) |     32     |      WS-DAN         |
|  [Stanford-Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)|     32     |      WS-DAN        |    
|  [FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)|     32     |      WS-DAN        |   


## Inspiration
The code is based on the [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim) Library.

## Preparing Datasets
Download and pre-process images and labels to tfrecords.

The convert_data.py will generate *./tfrecords* folder blow the provided $dataset_dir
* [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) Bird folder structure
```
-Bird
   └── Data
         └─── tfrecords
         └─── images.txt
         └─── image_class_labels.txt
         └─── train_test_split.txt
         └─── images
                 └─── ****.jpg
```
```
$ python convert_data.py --dataset_name=Bird --dataset_dir=./Bird/Data
```

* [Stanford-Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) folder structure
```
-Car
  └── Data
        └─── tfrecords
        └─── devkit
        |         └─── cars_train_annos.mat
        |         └─── cars_test_annos_withlabels.mat
        └─── cars_train
        |        └─── ****.jpg
        └─── cars_test
                 └─── ****.jpg
```
```
$ python convert_data.py --dataset_name=Car --dataset_dir=./Car/Data
```

* [FGVC-Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) folder structure
```
-Aircraft
    └── Data
          └─── tfrecords
          └─── fgvc-aircraft-2013b
                       └─── ***
```
```
$ python convert_data.py --dataset_name=Aircraft --dataset_dir=./Aircraft/Data
```

## Running training
### ImageNet pre-trained model
Download imagenet pre-trained model *[inception_v3.ckpt](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)* and put it blow folder *./pre_trained/*


```
DATASET="Bird"
TRAIN_DIR="./$DATASET/WS_DAN/TRAIN/ws_dan_part_32"
MODEL_PATH='./pre_trained/inception_v3.ckpt'

python train_sample.py --learning_rate=0.001 \
                            --dataset_name=$DATASET \
                            --dataset_dir="./$DATASET/Data/tfrecords" \
                            --train_dir=$TRAIN_DIR \
                            --checkpoint_path=$MODEL_PATH \
                            --max_number_of_steps=80000 \
                            --weight_decay=1e-5 \
                            --model_name='inception_v3_bap' \
                            --checkpoint_exclude_scopes="InceptionV3/bilinear_attention_pooling" \
                            --batch_size=12 \
                            --train_image_size=448 \
                            --num_clones=1 \
                            --gpus="3"\
                            --feature_maps="Mixed_6e"\
                            --attention_maps="Mixed_7a_b0"\
                            --num_parts=32
```
## Running testing
```
DATASET="Bird"
TRAIN_DIR="./$DATASET/WS_DAN/TRAIN/ws_dan_part_32"
TEST_DIR="./$DATASET/WS_DAN/TEST/ws_dan_part_32"

python eval_sample.py --checkpoint_path=$TRAIN_DIR \
                         --dataset_name=$DATASET \
                         --dataset_split_name='test' \
                         --dataset_dir="./$DATASET/Data/tfrecords" \
                         --eval_dir=$TEST_DIR \
                         --model_name='inception_v3_bap' \
                         --batch_size=16 \
                         --eval_image_size=448\
                         --gpus="2"\
                         --feature_maps="Mixed_6e"\
                         --attention_maps="Mixed_7a_b0"\
                         --num_parts=32
```

## Visualization
```
$ tensorboard --logdir=/path/to/model_dir --port=8081
```

## Contact
Email: yihouxiang@gmail.com

## License
MIT
