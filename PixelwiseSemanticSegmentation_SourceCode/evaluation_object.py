# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def cal_loss(logits, labels):
    loss_weight = np.array([
    0.25652730831033116,
    0.18528266685745448,
    4.396287575365375,
    0.1368693220338383,
    0.9184731310542199,
    0.38731986379829597,
    3.5330742906141994,
    1.8126852672146507,
    0.7246197983929721,
    5.855012980845159,
    8.136508447439535,
    1.0974099206087582
    ])

    labels = tf.to_int64(labels)
    loss, accuracy, prediction = weighted_loss(logits, labels, number_class=12, frequency=loss_weight)
    return loss, accuracy, prediction


def weighted_loss(logits, labels, number_class, frequency):
    label_flatten = tf.reshape(labels, [-1])
    label_onehot = tf.one_hot(label_flatten, depth=number_class)
    logits_reshape = tf.reshape(logits, [-1, number_class])
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=label_onehot, logits=logits_reshape,
                                                             pos_weight=frequency)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('loss', cross_entropy_mean)
    correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    tf.summary.scalar('accuracy', accuracy)

    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)


def normal_loss(logits, labels, number_class):
    label_flatten = tf.to_int64(tf.reshape(labels, [-1]))
    logits_reshape = tf.reshape(logits, [-1, number_class])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flatten, logits=logits_reshape,
                                                                   name='normal_cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('loss', cross_entropy_mean)
    correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    tf.summary.scalar('accuracy', accuracy)

    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)

def train_op(total_loss, opt):
    global_step = tf.Variable(0, trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        if (opt == "ADAM"):
            optimizer = tf.train.AdamOptimizer(0.001, epsilon=0.0001)
            print("Running with Adam Optimizer with learning rate:", 0.001)
        elif (opt == "SGD"):
            base_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(base_learning_rate, global_step, decay_steps=1000, decay_rate=0.0005)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            print("Running with Gradient Descent Optimizer with learning rate", 0.1)
        else:
            raise ValueError("Optimizer is not recognized")

        grads = optimizer.compute_gradients(total_loss, var_list=tf.trainable_variables())
        training_op = optimizer.apply_gradients(grads, global_step=global_step)

    return training_op, global_step