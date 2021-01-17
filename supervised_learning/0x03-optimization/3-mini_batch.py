#!/usr/bin/env python3
"""HYPERPARAMETER"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        batch_size=32,
        epochs=5,
        load_path="/tmp/model.ckpt",
        save_path="/tmp/model.ckpt"):
    """trains using mini-batch gradient descent"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        train_op = tf.get_collection('train_op')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(epochs + 1):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            loss_train, accuracy_train = sess.run((loss, accuracy),
                                                  feed_dict={
                x: X_train,
                y: Y_train
            })
            loss_valid, accuracy_valid = sess.run((loss, accuracy),
                                                  feed_dict={
                x: X_valid,
                y: Y_valid
            })
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(loss_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(loss_valid))
            print("\tValidation Accuracy: {}".format(
                accuracy_valid))
            nb_batches = int(X_train.shape[0] / batch_size)
            for j in range(nb_batches + 1):
                sess.run(train_op, feed_dict={
                    x: X_train[j * batch_size:(j + 1) * batch_size],
                    y: Y_train[j * batch_size:(j + 1) * batch_size]})
                if(j > 0) and (j % 100 == 0) and (j < nb_batches):
                    accuracy_step, loss_step = sess.run((accuracy, loss),
                                                          feed_dict={
                        x: X_train[j * batch_size:(j + 1) * batch_size],
                        y: Y_train[j * batch_size:(j + 1) * batch_size]})
                    print("\tStep: {}".format(j))
                    print("\t\tCost: {}".format(loss_step))
                    print("\t\tAccuracy:  {}".format(accuracy_step))
        return saver.save(sess, save_path)
