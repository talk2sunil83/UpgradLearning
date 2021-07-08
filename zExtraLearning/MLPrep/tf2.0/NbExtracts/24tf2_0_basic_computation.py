# -*- coding: utf-8 -*-
"""TF2.0 Basic Computation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DK-UHKAfjx_9aT8rbz35NipwHUKmKTlG
"""

# Commented out IPython magic to ensure Python compatibility.
# Install TensorFlow
import numpy as np
import tensorflow as tf
print(tf.__version__)

a = tf.constant(3.0)
b = tf.constant(4.0)
c = tf.sqrt(a**2 + b**2)
print("c:", c)

# if you use Python 3 f-strings it will print
# the tensor as a float
print(f"c: {c}")

# Get the Numpy version of a Tensor
c.numpy()

type(c.numpy())

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
print(f"b: {b}")
c = tf.tensordot(a, b, axes=[0, 0])
print(f"c: {c}")

a.numpy().dot(b.numpy())

A0 = np.random.randn(3, 3)
b0 = np.random.randn(3, 1)
c0 = A0.dot(b0)
print(f"c0: {c0}")

A = tf.constant(A0)
b = tf.constant(b0)
c = tf.matmul(A, b)
print(f"c: {c}")

# Broadcasting
A = tf.constant([[1, 2], [3, 4]])
b = tf.constant(1)
C = A + b
print(f"C: {C}")

# Element-wise multiplication
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[2, 3], [4, 5]])
C = A * B
print(f"C: {C}")