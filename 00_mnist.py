# Taken from https://www.tensorflow.org/tutorials/layers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Impor
import numpy as np
import tensorflow as tf
from eval import compute_map

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):

    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.identity(tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        tf.summary.scalar(name= 'training_loss', tensor = loss )
        summary_hook = tf.train.SummarySaverHook(
            100,
            output_dir='./models/01_mnist_0303_Train',
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, training_hooks= [summary_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    # print("-----------------"+str(np.shape(train_data)))
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./models/01_mnist_0303_Test")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)

    # Train the model


    print("train_input_fn ------------------> mnist_classifier.train")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter('./models/01_mnist_0303__mAp',
                                             sess.graph)
        for iteration in range(20):
            step_size = 100
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_data},
                y=train_labels,
                batch_size=100,
                num_epochs=None,
                shuffle=True)

            mnist_classifier.train(
                input_fn=train_input_fn,
                steps=step_size,
                hooks=[logging_hook])

            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data},
                y=eval_labels,
                num_epochs=1,
                shuffle=False)

            # Evaluate the model and print results
            eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
            accuracy = eval_results['accuracy']
            print("accuracy is {} ".format(accuracy))
            # pred = list(mnist_classifier.predict(input_fn=eval_input_fn))
            # pred = np.stack([p['probabilities'] for p in pred])
            # eval_weights = np.ones(eval_labels.shape[0])
            # AP = compute_map(eval_labels, pred, eval_weights, average=None)
            # print('01 Mnist Obtained {} mAP'.format(np.mean(AP)))
            # # print('02_Alexnet per class:')
            # # for cid, cname in enumerate(CLASS_NAMES):
            # #     print('{}: {}'.format(cname, _get_el(AP, cid)))
            # mAp = np.mean(AP)
            summary = tf.Summary(value=[tf.Summary.Value(tag="mAP", simple_value=accuracy)])
            train_writer.add_summary(summary, iteration*step_size)

        print("mnist_classifier.train ------------------> eval_input_fn")



if __name__ == "__main__":
    tf.app.run()
