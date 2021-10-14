import tensorflow as tf


def SubPixel1D(I, r):
  """One-dimensional subpixel upsampling layer
  Calls a tensorflow function that directly implements this functionality.
  We assume input has dim (batch, width, r)
  """
  with tf.name_scope('subpixel'):
    X = tf.transpose(I, [2,1,0]) # (r, w, b)
    X = tf.compat.v1.batch_to_space_nd(X, [r], [[0,0]]) # (1, r*w, b)
    X = tf.transpose(X, [2,1,0])
    return X