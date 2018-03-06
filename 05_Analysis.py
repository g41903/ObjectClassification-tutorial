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
    input_image_layer = []
    if img_num is not None:
        for img_idx in range(img_num):
            image = input_layer[img_idx,:]
            image = tf.random_crop(value = image, size = [227, 227, 3])
            image = tf.image.flip_left_right(image)
            image = tf.image.resize_image_with_crop_or_pad(image=image,target_height = 227, target_width = 227)
            input_image_layer.append(image)

        input_image_layer = tf.convert_to_tensor(input_image_layer, dtype=tf.float32)
    else:
        input_image_layer = input_layer
        print('img_num shape {}: input_layer is {} '.format(img_num, np.shape(input_layer.get_shape().as_list())))
        print("img_num is None")

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_image_layer,
        filters=96,
        kernel_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_size=[11, 11],
        strides=(4, 4),
        padding="valid",
        activation=tf.nn.relu)

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
        filters=384,
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
    pool5_list = pool5_shape.as_list()
    pool5_product = np.int32(pool5_list[1]*pool5_list[2]*pool5_list[3])
    pool5_flat = tf.reshape(pool5, [-1, pool5_product])

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
            output_dir='./models/05_VGG_NN_Pool5FC7_No_Pretrained_0303_Train',
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

    # Please uncomment if there is no images_dict
    '''
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
    '''
    imgs_dict = load_obj("images_dict")
    
    # pictures in diffeent classes
    labels_idx = ['000067','000015','000040','000069','000144','000054','000071','000049','000006','000025','000006','000018']
    eval_imgs = []
    for label_idx in labels_idx:
        eval_imgs.append(imgs_dict[label_idx])
    eval_imgs = np.asarray(eval_imgs)
    img_num = np.shape(eval_imgs)[0]
    eval_imgs_label = np.ones((img_num,20))
    eval_imgs_weights = np.ones((img_num,20))
    pool5_partial_test = nearest_neighbor_inference(eval_imgs,eval_imgs_label,eval_imgs_weights)

    pool5_list = load_obj("Final_No_Pretrained_pool5_list")
    pool5_flattened = np.reshape(pool5_list, [np.shape(pool5_list)[0], -1])

    pool5_partial_test_flattened = np.reshape(pool5_partial_test, [np.shape(pool5_partial_test)[0], -1])

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pool5_flattened)
    distances, indices = nbrs.kneighbors(pool5_partial_test_flattened)
    print('pool5_list {}: distances{} train_labels{}'.format(np.shape(pool5_list), distances, indices))

    return 0



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


    im1 = train_data[1, :, :, :]
    im2 = train_data[1, :, :, :]
    im3 = im1 * 0.5 + im2 * 0.5

    # img = Image.fromarray(im3)
    cv2.imwrite('color_img3.jpg', im3)
    cv2.imshow("image1", im3);
    cv2.waitKey();


def nearest_neighbor_inference(eval_imgs,eval_imgs_label,eval_imgs_weights):

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pascal_classifier = tf.estimator.Estimator(
            model_fn=partial(cnn_model_fn,
                             num_classes=20),
            model_dir="./models/05_VGG_NN_Pool5FC7_No_Pretrained_0303_Test")

        tensors_to_log = {"loss": "loss"}

        for iteration in range(1):

            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_imgs, "w": eval_imgs_weights},
                y=eval_imgs_label,
                batch_size=1,
                num_epochs=1,
                shuffle=False)
            pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
            pred_pool5 = np.stack([p['pool5'] for p in pred])
            pred_dense7 = np.stack([p['dense7'] for p in pred])
            print("pool5: {} dense7: {}".format(np.shape(pred_pool5),np.shape(pred_dense7)))
            print("------")

            save_obj(pred_pool5, "pool5_test")
            save_obj(pred_dense7, "dense7_test")
    return pred_pool5


def main():
    count = 0
    # print("Fast load pascal data----------------")
    # train_data = np.load(os.path.join(docs_dir, 'outfile_train_data.npy'))
    # train_labels = np.load(os.path.join(docs_dir, 'outfile_train_labels.npy'))
    # train_weights = np.load(os.path.join(docs_dir, 'outfile_train_weights.npy'))
    # eval_data = np.load(os.path.join(docs_dir, 'outfile_eval_data.npy'))
    # eval_labels = np.load(os.path.join(docs_dir, 'outfile_eval_labels.npy'))
    # eval_weights = np.load(os.path.join(docs_dir, 'outfile_eval_weights.npy'))
    # print('train_data {}: train_labels{} train_weights{}'.format(np.shape(train_data), np.shape(train_labels),
    #                                                              np.shape(train_weights)))
    # print("Finish load pascal data----------------")


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pascal_classifier = tf.estimator.Estimator(
            model_fn=partial(cnn_model_fn,
                             num_classes=train_labels.shape[1]),
            model_dir="./models/05_VGG_NN_Pool5FC7_No_Pretrained_0303_Test")

        tensors_to_log = {"loss": "loss"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)
        train_writer = tf.summary.FileWriter('./models/05_VGG_NN_Pool5FC7_No_Pretrained_0303_mAp',
                                             sess.graph)
        pool5_list = []
        dense7_list = []

        for iteration in range(1):
            # Train the model
            # train_input_fn = tf.estimator.inputs.numpy_input_fn(
            #     x={"x": train_data, "w": train_weights},
            #     y=train_labels,
            #     batch_size=100,
            #     num_epochs=None,
            #     shuffle=True)
            #
            # pascal_classifier.train(
            #     input_fn=train_input_fn,
            #     steps=100,
            #     hooks=[logging_hook])

            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data, "w": eval_weights},
                y=eval_labels,
                batch_size=1,
                num_epochs=1,
                shuffle=False)

            pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
            pred_pool5 = np.stack([p['pool5'] for p in pred])
            pred_dense7 = np.stack([p['dense7'] for p in pred])
            # dense7 shape = (4096,1)
            # dense7 = p['dense7']
            # # pool5 shape: (6, 6, 256)
            # pool5 = p['pool5']
            # pool5_list.append(pred_pool5)
            # dense7_list.append(pred_dense7)
            print("pool5: {} dense7: {}".format(np.shape(pred_pool5),np.shape(pred_dense7)))
            print("------")
            # rand_AP = compute_map(
            #    eval_labels, np.random.random(eval_labels.shape),
            #    eval_weights, average=None)
            # print('02_Alexnet Random AP: {} mAP'.format(np.mean(rand_AP)))
            # gt_AP = compute_map(
            #    eval_labels, eval_labels, eval_weights, average=None)
            # print('02_Alexnet GT AP: {} mAP'.format(np.mean(gt_AP)))
            # AP = compute_map(eval_labels, pred, eval_weights, average=None)
            # print('05_Analysis Obtained {} mAP'.format(np.mean(AP)))
            # print('02_Alexnet per class:')
            # for cid, cname in enumerate(CLASS_NAMES):
            #     print('{}: {}'.format(cname, _get_el(AP, cid)))
            # count += 1
            # scipy.misc.imsave('./features/conv_feature_outputs/dense7_{}.jpg'.format(count), pool5[:, :, 1:4])
            # scipy.misc.imsave('./features/conv_feature_outputs/dense7_{}.jpg'.format(count), dense7[:, :, 1:4])
            # mAp = np.mean(AP)
            # summary = tf.Summary(value=[tf.Summary.Value(tag="mAP", simple_value=mAp)])
            # train_writer.add_summary(summary)
            save_obj(pred_pool5, "Final_No_Pretrained_pool5_list")
            save_obj(pred_dense7, "Final_No_Pretrained_dense7_list")


        '''
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph('./models/pascal_model_scratch03-2/model.ckpt-21951.meta')
        model_test = new_saver.restore(sess, './models/pascal_model_scratch03-2/model.ckpt-21951')
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        '''

if __name__ == "__main__":
    # main()
    nearest_neigbor()