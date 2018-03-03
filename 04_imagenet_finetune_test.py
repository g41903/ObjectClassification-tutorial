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
docs_dir = os.path.expanduser('./data')

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
    input_layer = tf.reshape(features["x"], [-1, 256, 256, 3])
    img_num = input_layer.get_shape().as_list()[0]
    input_image_layer = input_layer

    '''
    if img_num is not None:
        for img_idx in range(img_num):
            image = input_layer[img_idx,:]
            image = tf.random_crop(value = image, size = [224, 224, 3])
            image = tf.image.flip_left_right(image)
            image = tf.image.resize_image_with_crop_or_pad(image=image,target_height = 224, target_width = 224)
            input_image_layer.append(image)

        input_image_layer = tf.convert_to_tensor(input_image_layer, dtype=tf.float32)
    else:
        input_image_layer = input_layer
        print('img_num shape {}: input_layer is {} '.format(img_num, np.shape(input_layer.get_shape().as_list())))
        print("img_num is None")
    '''

    # Convolutional Layer #1
    # init = tf.initializers.random_normal()
    # pad = 1
    conv1_1 = tf.layers.conv2d(
        inputs=input_image_layer,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    # pad = 1
    conv1_2 = tf.layers.conv2d(
        inputs=conv1_1,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)


    conv2_1 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)


    conv2_2 = tf.layers.conv2d(
        inputs= conv2_1,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

    conv3_1 = tf.layers.conv2d(
        inputs= pool2,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv3_2 = tf.layers.conv2d(
        inputs= conv3_1,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv3_3 = tf.layers.conv2d(
        inputs= conv3_2,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)


    conv4_1 = tf.layers.conv2d(
        inputs= pool3,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv4_2 = tf.layers.conv2d(
        inputs= conv4_1,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv4_3 = tf.layers.conv2d(
        inputs= conv4_2,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4_3, pool_size=[2, 2], strides=2)

    conv5_1 = tf.layers.conv2d(
        inputs= pool4,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv5_2 = tf.layers.conv2d(
        inputs= conv5_1,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    conv5_3 = tf.layers.conv2d(
        inputs= conv5_2,
        filters=512,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu)

    pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)

    # Dense Layer
    # pool5_shape = pool5.get_shape()
    # pool5_list = pool5_shape.as_list()
    # pool5_product = np.int32(pool5_list[1]*pool5_list[2]*pool5_list[3])
    # pool5_flatten = tf.reshape(pool5, [-1, pool5_product])

    dense6 = tf.layers.conv2d(inputs=pool5, filters=4096, padding="valid", kernel_size=[7, 7], strides=(2, 2), activation=tf.nn.relu)

    dropout6 = tf.layers.dropout(
        inputs=dense6, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)


    dense7 = tf.layers.conv2d( inputs= dropout6, filters= 4096, padding="valid",kernel_size=[1, 1], strides=(1, 1), activation=tf.nn.relu)

    dropout7 = tf.layers.dropout(inputs=dense7, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense8 = tf.layers.conv2d(inputs=dropout7, filters=4096, padding="valid", kernel_size=[1, 1], strides=(1, 1), activation=tf.nn.relu)

    dropout8 = tf.layers.dropout(inputs=dense8, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dropout8_flatten = tf.reshape(dropout8,[-1,4096])

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout8_flatten, units=20)


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

    loss = tf.identity(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,logits=logits))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # global_step = tf.Variable(0, trainable=False)
        grad_input = tf.gradients(loss,input_layer)
        grad_conv1_1 = tf.gradients(loss, conv1_1)
        grad_conv2_1 = tf.gradients(loss, conv2_1)
        grad_conv3_1 = tf.gradients(loss, conv3_1)
        grad_conv4_1 = tf.gradients(loss, conv4_1)
        grad_conv5_1 = tf.gradients(loss, conv5_1)
        grad_dense6 = tf.gradients(loss, dense6)
        grad_dense7 = tf.gradients(loss, dense7)

        starter_learning_rate = 0.001
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate= starter_learning_rate, global_step = global_step,
                                                   decay_steps = 100000, decay_rate= 0.5, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        #tf.summary()
        # print("Training")
        tf.summary.scalar(name= 'train_loss', tensor = loss )
        tf.summary.scalar(name= 'learning rate', tensor = learning_rate)
        tf.summary.histogram(name='grad_dense7', values=grad_input)
        tf.summary.histogram(name='grad_conv1_1', values= grad_conv1_1)
        tf.summary.histogram(name='grad_conv2_1', values=grad_conv2_1)
        tf.summary.histogram(name='grad_conv3_1', values=grad_conv3_1)
        tf.summary.histogram(name='grad_conv4_1', values=grad_conv4_1)
        tf.summary.histogram(name='grad_conv5_1', values=grad_conv5_1)
        tf.summary.histogram(name='grad_dense6', values=grad_dense6)
        tf.summary.histogram(name='grad_dense7', values=grad_dense7)

        tf.summary.image(name='image', tensor= input_layer)

        summary_hook = tf.train.SummarySaverHook(
            10,
            output_dir='./models/05_VGG_Pretrained_Classes_0303_Test',
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks = [summary_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



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

    data_dir = './data'
    '''
    train_data, train_labels, train_weights = load_pascal(
        data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        data_dir, split='test')

    # save files
    print("Save Fast load pascal data----------------")

    np.save(os.path.join(docs_dir, 'outfile_train_data'), train_data)
    np.save(os.path.join(docs_dir, 'outfile_train_labels'), train_labels)
    np.save(os.path.join(docs_dir, 'outfile_train_weights'), train_weights)
    np.save(os.path.join(docs_dir, 'outfile_eval_data'), eval_data)
    np.save(os.path.join(docs_dir, 'outfile_eval_labels'), eval_labels)
    np.save(os.path.join(docs_dir, 'outfile_eval_weights'), eval_weights)
    print("Finished Fast load pascal data----------------")
    '''



    pretrain_model_dict = {'vgg_16/conv1/conv1_1/biases': 'conv2d/bias',
                               'vgg_16/conv1/conv1_1/weights': 'conv2d/kernel',
                               'vgg_16/conv1/conv1_2/biases': 'conv2d_1/bias',
                               'vgg_16/conv1/conv1_2/weights': 'conv2d_1/kernel',
                               'vgg_16/conv2/conv2_1/biases': 'conv2d_2/bias',
                               'vgg_16/conv2/conv2_1/weights': 'conv2d_2/kernel',
                               'vgg_16/conv2/conv2_2/biases': 'conv2d_3/bias',
                               'vgg_16/conv2/conv2_2/weights': 'conv2d_3/kernel',
                               'vgg_16/conv3/conv3_1/biases': 'conv2d_4/bias',
                               'vgg_16/conv3/conv3_1/weights': 'conv2d_4/kernel',
                               'vgg_16/conv3/conv3_2/biases': 'conv2d_5/bias',
                               'vgg_16/conv3/conv3_2/weights': 'conv2d_5/kernel',
                               'vgg_16/conv3/conv3_3/biases': 'conv2d_6/bias',
                               'vgg_16/conv3/conv3_3/weights': 'conv2d_6/kernel',
                               'vgg_16/conv4/conv4_1/biases': 'conv2d_7/bias',
                               'vgg_16/conv4/conv4_1/weights': 'conv2d_7/kernel',
                               'vgg_16/conv4/conv4_2/biases': 'conv2d_8/bias',
                               'vgg_16/conv4/conv4_2/weights': 'conv2d_8/kernel',
                               'vgg_16/conv4/conv4_3/biases': 'conv2d_9/bias',
                               'vgg_16/conv4/conv4_3/weights': 'conv2d_9/kernel',
                               'vgg_16/conv5/conv5_1/biases': 'conv2d_10/bias',
                               'vgg_16/conv5/conv5_1/weights': 'conv2d_10/kernel',
                               'vgg_16/conv5/conv5_2/biases': 'conv2d_11/bias',
                               'vgg_16/conv5/conv5_2/weights': 'conv2d_11/kernel',
                               'vgg_16/conv5/conv5_3/biases': 'conv2d_12/bias',
                               'vgg_16/conv5/conv5_3/weights': 'conv2d_12/kernel',
                               'vgg_16/fc6/biases': 'conv2d_13/bias',
                               'vgg_16/fc6/weights': 'conv2d_13/kernel',
                               'vgg_16/fc7/biases': 'conv2d_14/bias',
                               'vgg_16/fc7/weights': 'conv2d_14/kernel'}

    class _LossCheckerHook(tf.train.SessionRunHook):
        def begin(self):
            # load checkpoint
            checkpoint_dir = './models/vgg_16.ckpt'
            tf.train.init_from_checkpoint(checkpoint_dir, pretrain_model_dict)

    print("Fast load pascal data----------------")
    train_data = np.load(os.path.join(docs_dir, 'outfile_train_data.npy'))
    train_labels = np.load(os.path.join(docs_dir, 'outfile_train_labels.npy'))
    train_weights = np.load(os.path.join(docs_dir, 'outfile_train_weights.npy'))
    eval_data = np.load(os.path.join(docs_dir, 'outfile_eval_data.npy'))
    eval_labels = np.load(os.path.join(docs_dir, 'outfile_eval_labels.npy'))
    eval_weights = np.load(os.path.join(docs_dir, 'outfile_eval_weights.npy'))
    print('train_data {}: train_labels{} train_weights{}'.format(np.shape(train_data), np.shape(train_labels),np.shape(train_weights)))
    print("Finish load pascal data----------------")

    hook_object = _LossCheckerHook()

    init = tf.global_variables_initializer()

    # saver = tf.train.Saver()
    # with tf.Session as sess:
    #     saver.restore(sess, "./models/vgg_16.ckpt")
    #     print("Model restored.")
    #     # Check the values of the variables
    #     print("v1 : %s" % vgg_16/conv5/conv5.eval())
    #     print("v2 : %s" % v2.eval())

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter('./models/05_VGG_Pretrained_Classes_0303_mAp',
                                             sess.graph)
        pascal_classifier = tf.estimator.Estimator(
            model_fn=partial(cnn_model_fn,
                             num_classes=train_labels.shape[1]),
            model_dir="./models/05_VGG_Pretrained_Classes_0303_Train")

        tensors_to_log = {"loss": "loss"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)


        # Train the model
        for iteration in range(40):
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_data, "w": train_weights},
                y=train_labels,
                batch_size=10,
                num_epochs=None,
                shuffle=False)

            pascal_classifier.train(
                input_fn=train_input_fn,
                steps=100,
                hooks=[logging_hook, hook_object])

            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data, "w": eval_weights},
                y=eval_labels,
                batch_size=32,
                num_epochs=1,
                shuffle=False)

            pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
            pred = np.stack([p['probabilities'] for p in pred])
            #rand_AP = compute_map(
            #    eval_labels, np.random.random(eval_labels.shape),
            #    eval_weights, average=None)
            #print('03_VGG Random AP: {} mAP'.format(np.mean(rand_AP)))
            #gt_AP = compute_map(
            #    eval_labels, eval_labels, eval_weights, average=None)
            #print('03_VGG GT AP: {} mAP'.format(np.mean(gt_AP)))
            AP = compute_map(eval_labels, pred, eval_weights, average=None)
            print('03_VGG Obtained {} mAP'.format(np.mean(AP)))
            print('03_VGG per class:')
            for cid, cname in enumerate(CLASS_NAMES):
               print('{}: {}'.format(cname, _get_el(AP, cid)))

            mAp = np.mean(AP)
            summary = tf.Summary(value=[tf.Summary.Value(tag="mAP",simple_value=mAp)])
            train_writer.add_summary(summary)
        train_writer.flush()



if __name__ == "__main__":
    main()
