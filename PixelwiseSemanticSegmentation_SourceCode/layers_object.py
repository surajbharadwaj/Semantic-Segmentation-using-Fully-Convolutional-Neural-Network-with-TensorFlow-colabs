# -*- coding: utf-8 -*-
import tensorflow as tf
import math


def max_pool(inputs, name):
    with tf.variable_scope(name) as scope:
        value, index = tf.nn.max_pool_with_argmax(tf.to_double(inputs), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)
    return tf.to_float(value), index, inputs.get_shape().as_list()


def conv_layer(input_map, name, shape, use_vgg=False, vgg_param_dict=None):
    def get_conv_filter(val_name):
        return vgg_param_dict[val_name][0]

    def get_biases(val_name):
        return vgg_param_dict[val_name][1]

    with tf.variable_scope(name) as scope:
        if use_vgg:
            init = tf.constant_initializer(get_conv_filter(scope.name))
            filt = tf.get_variable('weights', shape=shape, initializer=init)
        else:
            filt = tf.get_variable('weights', shape=shape, initializer=initialization(shape[0], shape[2]))
        conv = tf.nn.conv2d(input_map, filt, [1, 1, 1, 1], padding='SAME')
        if use_vgg:
            conv_biases_init = tf.constant_initializer(get_biases(scope.name))
            conv_biases = tf.get_variable('biases', shape=shape[3], initializer=conv_biases_init)
        else:
            conv_biases = tf.get_variable('biases', shape=shape[3], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, conv_biases)
        conv_out = tf.nn.relu(batch_norm(bias, scope))
        tf.summary.histogram("weights", filt)
        tf.summary.histogram("biases", conv_biases)
        tf.summary.histogram("activations", conv_out)
        
    return conv_out


def batch_norm(bias_input, scope):
    with tf.variable_scope(scope.name) as scope:
        return  tf.contrib.layers.batch_norm(bias_input, is_training=False, center=False, scope=scope)

def up_sampling(pool, ind, output_shape, batch_size, name=None):
    with tf.variable_scope(name):
        pool_ = tf.reshape(pool, [-1])
        batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), [tf.shape(pool)[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [-1, 1])
        ind_ = tf.reshape(ind, [-1, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
        ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
        return ret

def initialization(k, c):
    """
    k is the filter size
    c is the number of input channels in the filter tensor
    """
    std = math.sqrt(2. / (k ** 2 * c))
    return tf.truncated_normal_initializer(stddev=std)