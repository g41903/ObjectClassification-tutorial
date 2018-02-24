from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import glob, os
import os.path as osp
from PIL import Image
from functools import partial

from eval import compute_map
import models

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function


def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that 
            are ambiguous.
    """
    # Wrote this function
    img_size = 256,256
    images = []
    for infile in glob.glob("./VOCdevkit/VOC2007/JPEGImages/*.jpg"):
        # reshape the images to 256*256*3
        try:
            im = Image.open(infile)
            resized_img = im.resize((256, 256), Image.ANTIALIAS)
            resized_arr = np.array(resized_img)
            images.append(resized_arr)
            # np.asarray(images)
            # print(np.shape(images))
            print(type(images))
        except IOError:
            print("Error")

        # convert list into ndarray
        images = np.asarray(images)
        images_size = np.shape(images)
        images_num = images_size[0]
        # label_mat: 2d array, each annotation file is one label_col, multiple label_col mean multiple annotation files
        label_mat = np.array((images_num,21))
        label_col = []

        for filename in os.listdir("./VOCdevkit/VOC2007/ImageSets/Main/"):

            if filename.endswith("test.txt"):
                # print(os.path.join(directory, filename))
                print(filename)
                with open("./VOCdevkit/VOC2007/ImageSets/Main/"+filename) as fp:
                    line = fp.readline()
                    cnt = 1
                    while line:
                        print("Line {}: {}".format(cnt, line.strip()))
                        line = fp.readline()
                        label_col.append(line)
                        cnt += 1
                np.vstack(label_mat,label_col)
                continue
            else:
                continue
        print(np.shape(label_mat))


                # with open("./VOCdevkit/VOC2007/ImageSets/Main/bird_test.txt") as fp:
        #     line = fp.readline()
        #     cnt = 1
        #     while line:
        #         print("Line {}: {}".format(cnt, line.strip()))
        #         line = fp.readline()
        #         cnt += 1


    # for img_idx in range():


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def main():
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="/tmp/pascal_model_scratch")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    pascal_classifier.train(
        input_fn=train_input_fn,
        steps=NUM_ITERS,
        hooks=[logging_hook])
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pred = np.stack([p['probabilities'] for p in pred])
    rand_AP = compute_map(
        eval_labels, np.random.random(eval_labels.shape),
        eval_weights, average=None)
    print('Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = compute_map(
        eval_labels, eval_labels, eval_weights, average=None)
    print('GT AP: {} mAP'.format(np.mean(gt_AP)))
    AP = compute_map(eval_labels, pred, eval_weights, average=None)
    print('Obtained {} mAP'.format(np.mean(AP)))
    print('per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, _get_el(AP, cid)))


if __name__ == "__main__":
    main()
