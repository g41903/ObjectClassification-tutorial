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

# extra
from tempfile import TemporaryFile
import pickle



# import models

# Run the code, type the command in the terminal
# python 01_pascal.py /home/teame-predict/Documents/ernie/ObjectClassification-tutorial

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
docs_dir = os.path.expanduser('~/Documents/ernie')

def save_obj(obj, name ):
    with open(docs_dir+'/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(docs_dir+'/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def cnn_model_fn(features, labels, mode, num_classes=20):
    # Write this function
    """Model function for CNN."""
    # Input Layer
    print('{}: {}'.format(np.shape(features), np.shape(labels)))
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])

    # filter1 = tf.random_normal(shape=[11,11,3,96])
    # conv1 = tf.layers.conv2d(
    #     inputs=input_layer,
    #     filters=32,
    #     kernel_size=[5, 5],
    #     padding="same",
    #     activation=tf.nn.relu)
    # Convolutional Layer #1

    # init = tf.initializers.random_normal()
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_size=[11, 11],
        strides=(4, 4),
        padding="valid",
        activation=tf.nn.relu)
    # bias_layer_1 = tf.nn.bias_add(conv1, biases_1_array)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)


    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        #filters=tf.Variable(tf.random_normal([5, 5, 96, 256])),
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_size=[5, 5],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        #filters=tf.random_normal(shape=[3, 3, 256, 384]),
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #4 and Pooling Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=256,
        #filters=tf.random_normal(shape=[3, 3, 384, 384]),
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    # Convolutional Layer #2 and Pooling Layer #2
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        #filters=tf.random_normal(shape=[3, 3, 384, 256]),
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same")
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

    # Dense Layer
    pool5_shape = pool5.get_shape()
    print(".....................................")
    print("pool5_shape is {}".format(np.shape(pool5_shape)))

    pool5_flat = tf.reshape(pool5, [-1, 6*6*256])

    dense6 = tf.layers.dense(inputs=pool5_flat, units=4096,activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),bias_initializer=tf.zeros_initializer(),)
    dropout6 = tf.layers.dropout(
        inputs=dense6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)


    dense7 = tf.layers.dense(inputs=dropout6, units= 4096, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer(),)
    dropout7 = tf.layers.dropout(
        inputs=dense7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout7, units=20)


    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        # "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    # Calculate Loss (for both TRAIN and EVAL modes)
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    print("--------------------------------------------------")
    print("labels is {} and logits {}: and type".format(np.shape(labels), np.shape(logits)))

    loss = tf.identity(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,logits=logits))
    '''
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100000, 0.96, staircase=True)
    # Passing global_step to minimize() will increment it at each step.
    learning_step = (
        tf.train.GradientDescentOptimizer(learning_rate)
            .minimize(...
    my
    loss..., global_step = global_step)
    )
    '''

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate= starter_learning_rate, global_step = global_step,
                                                   decay_steps = 100000, decay_rate= 0.5, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



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
    # idx = 0
    # if idx >20:
    #     idx+=1
    #     break
    """
    print("Begin Load Images ------------------------------------")
    images = []
    # images_dict -> key: img_file_idx, value: rgb image ndarray (256*256*3)
    images_dict = {}
    # count
    for infile in glob.glob("./VOCdevkit/VOC2007/JPEGImages/*.jpg"):
        # reshape the images to 256*256*3
        file, ext = os.path.splitext(infile)
        file_idx = file[-6:]

        try:
            im = Image.open(infile)
            resized_img = im.resize((256, 256), Image.ANTIALIAS)
            resized_arr = np.array(resized_img)
            images_dict[file_idx] = resized_arr.astype(np.float32)
        except IOError:
            print("Error")

    save_obj(images_dict,"images_dict")
    """
    # label_mat: 2d array, each annotation file is one label_col, multiple label_col mean multiple annotation files
    label_mat = []
    weight_mat = []
    image_mat = []

    images_dict = load_obj("images_dict")
    print("Return Load Images ------------------------------------")

    idx= 0
    line_limit =9960
    # for filename in os.listdir("./VOCdevkit/VOC2007/ImageSets/Main/"):
    for filename in enumerate(CLASS_NAMES):

        with open("./VOCdevkit/VOC2007/ImageSets/Main/"+filename[1] +"_"+split+".txt") as fp:
            print(fp)
            image_mat = []
            label_col = []
            weight_col = []
            line = fp.readline()
            cnt = 1
            while line:

                label_idx = line.strip()[:-3]
                #if (int(label_idx)>line_limit or int(label_idx)<=0):
                #    break
                try:
                    # print("Line {}: {}".format(label_idx, type(label_idx)))
                    # Be aware!! '000005 ' is different from '000005', there is a space in the first string!!!
                    # label_idx = '000005 ' label_idx[:-1]='000005'
                    image_mat.append(images_dict[label_idx])
                except IOError:
                    print("Error Line {}: {}".format(label_idx, type(label_idx)))

                label_flag = int(line.strip()[-2:])

                if label_flag is 0 or label_flag is -1:
                    label_col.append(np.int32(0))
                else:
                    label_col.append(np.int32(1))

                if label_flag is 1 or label_flag is -1:
                    weight_col.append(np.int32(1))
                else:
                    weight_col.append(np.int32(0))

                line = fp.readline()
                cnt += 1
            np_label_col = np.asarray(label_col)
            label_mat.append(np_label_col)
            # print(np.shape(label_mat))
            np_weight_col = np.asarray(weight_col)
            weight_mat.append(np_weight_col)


    print("********************")
    # print('image_mat {}: label_mat {}'.format(np.shape(image_mat), np.shape(label_mat)))
    np_image_mat = np.asarray(image_mat)
    np_label_mat = np.asarray(label_mat)
    np_weight_mat = np.asarray(weight_mat)
    # print('np_image_mat {}: np_label_mat {}'.format(np.shape(np_image_mat), np.shape(np_label_mat)))
    np_trans_label_mat = np_label_mat.transpose()
    np_trans_weight_mat = np_weight_mat.transpose()
    # print(np.shape(np_label_mat))
    # print(np.shape(np_weight_mat))
    print('np_trans_label_mat {}: np_trans_weight_mat {}'.format(np.shape(np_trans_label_mat), np.shape(np_trans_weight_mat)))
    print("Return Load Weights and Labels ------------------------------------")
    return np_image_mat, np_trans_label_mat, np_trans_weight_mat


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='/home/teame-predict/Documents/ernie/ObjectClassification-tutorial',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    print("args")
    print("#####################################")
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def main():
    # args = parse_args()
    # print(args)
    # Load training and eval data

    # outfile_train_data = TemporaryFile()
    # outfile_train_labels = TemporaryFile()
    # outfile_train_weights = TemporaryFile()
    # outfile_eval_data = TemporaryFile()
    # outfile_eval_labels = TemporaryFile()
    # outfile_eval_weights = TemporaryFile()
    data_dir = '/home/teame-predict/Documents/ernie/ObjectClassification-tutorial'

    # train_data, train_labels, train_weights = load_pascal(
    #     data_dir, split='trainval')
    # eval_data, eval_labels, eval_weights = load_pascal(
    #     data_dir, split='test')
    #
    # # save files
    # print("Save Fast load pascal data----------------")
    #
    # np.save(os.path.join(docs_dir, 'outfile_train_data'), train_data)
    # np.save(os.path.join(docs_dir, 'outfile_train_labels'), train_labels)
    # np.save(os.path.join(docs_dir, 'outfile_train_weights'), train_weights)
    # np.save(os.path.join(docs_dir, 'outfile_eval_data'), eval_data)
    # np.save(os.path.join(docs_dir, 'outfile_eval_labels'), eval_labels)
    # np.save(os.path.join(docs_dir, 'outfile_eval_weights'), eval_weights)
    # print("Finished Fast load pascal data----------------")


    print("Fast load pascal data----------------")
    train_data = np.load(os.path.join(docs_dir, 'outfile_train_data.npy'))
    train_labels = np.load(os.path.join(docs_dir, 'outfile_train_labels.npy'))
    train_weights = np.load(os.path.join(docs_dir, 'outfile_train_weights.npy'))
    eval_data = np.load(os.path.join(docs_dir, 'outfile_eval_data.npy'))
    eval_labels = np.load(os.path.join(docs_dir, 'outfile_eval_labels.npy'))
    eval_weights = np.load(os.path.join(docs_dir, 'outfile_eval_weights.npy'))
    print('train_data {}: train_labels{} train_weights{}'.format(np.shape(train_data), np.shape(train_labels),np.shape(train_weights)))
    print("Finish load pascal data----------------")

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="./models/pascal_model_scratch02")

    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=50,
        num_epochs=None,
        shuffle=True)
    pascal_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
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
    print('02_Alexnet Random AP: {} mAP'.format(np.mean(rand_AP)))
    gt_AP = compute_map(
        eval_labels, eval_labels, eval_weights, average=None)
    print('02_Alexnet GT AP: {} mAP'.format(np.mean(gt_AP)))
    AP = compute_map(eval_labels, pred, eval_weights, average=None)
    print('02_Alexnet Obtained {} mAP'.format(np.mean(AP)))
    print('02_Alexnet per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, _get_el(AP, cid)))


if __name__ == "__main__":
    main()
