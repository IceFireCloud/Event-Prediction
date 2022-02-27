#Modified for GAN.
#Last modified: 20/7/30
#Working: yes

import tensorflow as tf
import sys

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        lrelu5 = lrelu(tf.layers.batch_normalization(conv5, training=isTrain), 0.2)

        # 1st hidden layer
        conv6 = tf.layers.conv2d(lrelu5, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu6 = lrelu(conv6, 0.2)

        # 2nd hidden layer
        conv7 = tf.layers.conv2d(lrelu6, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu7 = lrelu(tf.layers.batch_normalization(conv7, training=isTrain), 0.2)

        # 3rd hidden layer
        conv8 = tf.layers.conv2d(lrelu7, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu8 = lrelu(tf.layers.batch_normalization(conv8, training=isTrain), 0.2)

        # 4th hidden layer
        conv9 = tf.layers.conv2d(lrelu8, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu9 = lrelu(tf.layers.batch_normalization(conv9, training=isTrain), 0.2)

        conv10 = tf.layers.conv2d(lrelu9, 1, [4, 4], strides=(1, 1), padding='valid')
        o = tf.nn.tanh(conv10)

        return o

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [1, 1], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(conv5)
        return o, conv5

print('model.py')
