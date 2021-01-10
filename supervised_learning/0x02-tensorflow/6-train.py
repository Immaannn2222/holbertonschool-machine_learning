#!/usr/bin/env python3
"""Tensorflow project"""
import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        layer_sizes,
        activations,
        alpha,
        iterations,
        save_path="/tmp/model.ckpt"):
    """builds, trains, saves a neural network classifier"""

    classes = Y_train.shape[1]
    nx = X_train.shape[1]
    x, y = create_placeholders(nx, classes)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    acc = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', acc)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    train = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as s:
        s.run(init)
        for i in range(iterations + 1):
            cost_train, ac_train = s.run((loss, acc), feed_dict={
                x: X_train,
                y: Y_train
            })
            cost_valid, ac_valid = s.run((loss, acc), feed_dict={
                x: X_valid,
                y: Y_valid
            })
            if i % 100 == 0 or i == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(ac_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(ac_valid))
            if(i != iterations):
                s.run(train, {x: X_train, y: Y_train})
        return saver.save(s, save_path)
