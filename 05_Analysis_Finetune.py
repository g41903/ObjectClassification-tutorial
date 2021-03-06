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
from PIL import Image

# extra
from tempfile import TemporaryFile
import pickle
import scipy.misc
import cv2
import random
from random import shuffle
from sklearn.neighbors import NearestNeighbors

docs_dir = os.path.expanduser('./data')


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

    # Data Augmentation
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
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor"),
        "pool5": pool5,
        "dense7": dense7
    }

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,logits=logits))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        starter_learning_rate = 0.001
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate= starter_learning_rate, global_step = global_step,
                                                   decay_steps = 100000, decay_rate= 0.5, staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        tf.summary.scalar(name= 'training_loss', tensor = loss )
        tf.summary.scalar(name= 'learning rate', tensor = learning_rate)
        tf.summary.image(name='image', tensor= input_layer)

        summary_hook = tf.train.SummarySaverHook(
            1000,
            output_dir='./models/05_VGG_NN_Pool5FC7_Pretrained_0303_Train',
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

    # Add evaluation metrics (for EVAL mode)
    accuracy  = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])

    eval_metric_ops = {
        "accuracy": accuracy}

    if mode == tf.estimator.ModeKeys.EVAL:
        tf.summary.scalar('accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def conv_1_filter():

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('./models/05_Pascal_AlexNet_0303_Test/model.ckpt-6000.meta')
    model_test = new_saver.restore(sess,'./models/05_Pascal_AlexNet_0303_Test/model.ckpt-6000')
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    tf.global_variables_initializer()

    # 5-1 pool5 features from the AlexNet
    # w : [11* 11* 3 * 96]


    pil_images = []
    # save 96 images in the file
    # convert from ndarayy to PIL image

    for i in range(96):
        # all the layers' tensor
        w = sess.run(all_vars)
        # w[1]: get the tensor of the first layer 11*11*3*96
        arr = w[1][:, :, :, i]
        # img = Image.fromarray(arr)
        scipy.misc.imsave('img{}.jpg'.format(i), arr)
        im = scipy.misc.imread('img{}.jpg'.format(i))
        pil_images.append(im)

    print("Finish saving feature map")

    height = sum(image.shape[0] for image in pil_images)
    width = max(image.shape[1] for image in pil_images)
    output = np.zeros((height, width, 3))

    y = 0
    for image in pil_images:
        h, w, d = image.shape
        output[y:y + h, 0:w] = image
        y += h

    cv2.imwrite("pool5_feature_map_6000.jpg", output)

def nearest_neigbor():

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('./models/02_Pascal_AlexNet_0302_Test/model.ckpt-7000.meta')
    model_test = new_saver.restore(sess,'./models/02_Pascal_AlexNet_0302_Test/model.ckpt-7000')
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    tf.global_variables_initializer()

    # for i in range(96):
    #     # all the layers' tensor
    #     w = sess.run(all_vars)
    #     # w[1]: get the tensor of the first layer 11*11*3*96
    #     arr = w[1][:, :, :, i]

    return 0

    # concatenate 96 images together to 12 * 8 image grid


    # widths, heights = zip(*(i.size for i in pil_images))
    # total_width = sum(widths)
    # max_height = max(heights)
    # new_im = Image.new('RGB', (total_width, max_height))
    # new_im = Image.new('RGB', ((88,88),(132,132)))
    # x_offset = 0
    # y_offset = 0
    #
    # for i in range(12):
    #     for j in range(8):
    #         im = pil_images[i*j+j]
    #         new_im.paste(im, (x_offset, y_offset))
    #         x_offset += 11
    #     y_offset += 11
    # new_im.save("Pool5.jpg")

def mixup():

    train_data = np.load(os.path.join(docs_dir, 'outfile_train_data.npy'))
    train_labels = np.load(os.path.join(docs_dir, 'outfile_train_labels.npy'))
    train_weights = np.load(os.path.join(docs_dir, 'outfile_train_weights.npy'))
    eval_data = np.load(os.path.join(docs_dir, 'outfile_eval_data.npy'))
    eval_labels = np.load(os.path.join(docs_dir, 'outfile_eval_labels.npy'))
    eval_weights = np.load(os.path.join(docs_dir, 'outfile_eval_weights.npy'))

    ind_list = [i for i in range(5011)]
    rand_list = random.shuffle(ind_list)
    ind_arr = np.asarray(ind_list)
    ind_flip = np.flip(ind_arr,0)
    train_rand = train_data[ind_flip,:,:,:]
    label_rand = train_labels[ind_flip,:]

    alpha = 0.3
    train_final = []
    label_final = []
    for i in range(train_data.shape[0]):
        train_final.append(alpha * train_data[i,:,:,:] + (1 - alpha) * train_rand[i,:,:,:])
        label_final.append(alpha * train_labels[i,:] + (1 - alpha) * label_rand[i,:])

    # train_final = alpha * train_data+(1-alpha) * train_rand
    # label_final = alpha * eval_data + (1-alpha) * label_rand

    # label_final = alpha * eval_data + (1 - alpha) * label_rand
    # label_final = alpha * eval_data + (1 - alpha) * label_rand

    im1 = train_data[1, :, :, :]
    im2 = train_data[1, :, :, :]
    im3 = im1 * 0.5 + im2 * 0.5

    # img = Image.fromarray(im3)
    cv2.imwrite('color_img3.jpg', im3)
    cv2.imshow("image1", im3);
    cv2.waitKey();


def main():
    count = 0
    print("Fast load pascal data----------------")
    train_data = np.load(os.path.join(docs_dir, 'outfile_train_data.npy'))
    train_labels = np.load(os.path.join(docs_dir, 'outfile_train_labels.npy'))
    train_weights = np.load(os.path.join(docs_dir, 'outfile_train_weights.npy'))
    eval_data = np.load(os.path.join(docs_dir, 'outfile_eval_data.npy'))
    eval_labels = np.load(os.path.join(docs_dir, 'outfile_eval_labels.npy'))
    eval_weights = np.load(os.path.join(docs_dir, 'outfile_eval_weights.npy'))
    print('train_data {}: train_labels{} train_weights{}'.format(np.shape(train_data), np.shape(train_labels),
                                                                 np.shape(train_weights)))
    print("Finish load pascal data----------------")

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

    hook_object = _LossCheckerHook()

    # Mix up is the following block
    '''
    ind_list = [i for i in range(5011)]
    rand_list = random.shuffle(ind_list)
    ind_arr = np.asarray(ind_list)
    ind_flip = np.flip(ind_arr,0)
    train_rand = train_data[ind_flip,:,:,:]
    label_rand = train_labels[ind_flip,:]

    alpha = 0.3
    train_final = []
    label_final = []
    for i in range(train_data.shape[0]):
        train_final.append(alpha * train_data[i,:,:,:] + (1 - alpha) * train_rand[i,:,:,:])
        label_final.append(alpha * train_labels[i,:] + (1 - alpha) * label_rand[i,:])

    print("train_final shape is {}".format(np.shape(train_final)))
    print("train_label shape is {}".format(np.shape(label_final)))
    train_final = np.asarray(train_final)
    label_final = np.asarray(label_final)
    '''
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pascal_classifier = tf.estimator.Estimator(
            model_fn=partial(cnn_model_fn,
                             num_classes=train_labels.shape[1]),
            model_dir="./models/05_VGG_NN_Pool5FC7_Pretrained_0303_Test")

        tensors_to_log = {"loss": "loss"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)
        train_writer = tf.summary.FileWriter('./models/05_VGG_NN_Pool5FC7_Pretrained_0303_mAp',
                                             sess.graph)
        pool5_list = []
        dense7_list = []

        for iteration in range(50):
            # Train the model
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_data, "w": train_weights},
                y=train_labels,
                batch_size=10,
                num_epochs=None,
                shuffle=True)

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
            # dense7 shape = (4096,1)
            dense7 = p['dense7']
            # pool5 shape: (6, 6, 256)
            pool5 = p['pool5']
            pool5_list.append(pool5)
            dense7_list.append(dense7)

            # print("dense7: {} pool5: {}".format(np.shape(dense7),np.shape(pool5)))
            # rand_AP = compute_map(
            #    eval_labels, np.random.random(eval_labels.shape),
            #    eval_weights, average=None)
            # print('02_Alexnet Random AP: {} mAP'.format(np.mean(rand_AP)))
            # gt_AP = compute_map(
            #    eval_labels, eval_labels, eval_weights, average=None)
            # print('02_Alexnet GT AP: {} mAP'.format(np.mean(gt_AP)))
            AP = compute_map(eval_labels, pred, eval_weights, average=None)
            print('05_Analysis Obtained {} mAP'.format(np.mean(AP)))
            # print('02_Alexnet per class:')
            # for cid, cname in enumerate(CLASS_NAMES):
            #     print('{}: {}'.format(cname, _get_el(AP, cid)))
            # count += 1
            # scipy.misc.imsave('./features/conv_feature_outputs/dense7_{}.jpg'.format(count), pool5[:, :, 1:4])
            # scipy.misc.imsave('./features/conv_feature_outputs/dense7_{}.jpg'.format(count), dense7[:, :, 1:4])
            mAp = np.mean(AP)
            summary = tf.Summary(value=[tf.Summary.Value(tag="mAP", simple_value=mAp)])
            train_writer.add_summary(summary)
        save_obj(dense7_list, "Pretrained_dense7_list")
        save_obj(pool5_list, "Pretrained_pool5_list")

    '''
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('./models/pascal_model_scratch03-2/model.ckpt-21951.meta')
    model_test = new_saver.restore(sess, './models/pascal_model_scratch03-2/model.ckpt-21951')
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    '''

if __name__ == "__main__":
    main()


'''
import cv2
import numpy
import glob
import os

dir = "." # current directory
ext = ".jpg" # whatever extension you want

pathname = os.path.join(dir, "*" + ext)
images = [cv2.imread(img) for img in glob.glob(pathname)]

height = sum(image.shape[0] for image in images)
width = max(image.shape[1] for image in images)
output = numpy.zeros((height,width,3))

y = 0
for image in images:
    h,w,d = image.shape
    output[y:y+h,0:w] = image
    y += h

cv2.imwrite("test.jpg", output)


'''