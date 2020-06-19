# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:17:58 2017

@author: Balázs Hidasi
"""

import tensorflow as tf



def gpu_diag_wide(X):
    E = tf.eye(X.get_shape().as_list()[0], X.get_shape().as_list()[1])
    return tf.reduce_sum(X * E, axis=1)


def gpu_diag_tall(X):
    E = tf.eye(X.get_shape().as_list()[0], X.get_shape().as_list()[1])
    return tf.reduce_sum(X * E, axis=0)