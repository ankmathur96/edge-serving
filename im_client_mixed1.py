import requests
import json
import tensorflow as tf
import numpy as np
import ops
import scopes
import datetime

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999
MODEL_VERSION = 1
dropout_keep_prob=0.8
num_classes=1000
is_training=False

batch_norm_params = {
      # Decay for the moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }

t1 = datetime.datetime.now()
with tf.Session() as sess:
  images = tf.random_uniform((1, 299,299,3), 0, 255)
  with scopes.arg_scope([ops.conv2d, ops.fc], weight_decay=0.00004):
      with scopes.arg_scope([ops.conv2d],
                          stddev=0.1,
                          activation=tf.nn.relu,
                          batch_norm_params=batch_norm_params):
        end_points = {}
        with tf.name_scope(None, 'inception_v3', [images]):
          with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                                is_training=is_training):
            with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                  stride=1, padding='VALID'):
              # 299 x 299 x 3
              end_points['conv0'] = ops.conv2d(images, 32, [3, 3], stride=2,
                                               scope='conv0')
              # 149 x 149 x 32
              end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3],
                                               scope='conv1')
              # 147 x 147 x 32
              end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3],
                                               padding='SAME', scope='conv2')
              # 147 x 147 x 64
              end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                                 stride=2, scope='pool1')
              # 73 x 73 x 64
              end_points['conv3'] = ops.conv2d(end_points['pool1'], 80, [1, 1],
                                               scope='conv3')
              # 73 x 73 x 80.
              end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3],
                                               scope='conv4')
              # 71 x 71 x 192.
              end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                                 stride=2, scope='pool2')
              # 35 x 35 x 192.
              net = end_points['pool2']

            # Inception blocks
            with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                                  stride=1, padding='SAME'):
              # mixed: 35 x 35 x 256.
              with tf.variable_scope('mixed_35x35x256a'):
                with tf.variable_scope('branch1x1'):
                  branch1x1 = ops.conv2d(net, 64, [1, 1])
                with tf.variable_scope('branch5x5'):
                  branch5x5 = ops.conv2d(net, 48, [1, 1])
                  branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                with tf.variable_scope('branch3x3dbl'):
                  branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                  branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                  branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                with tf.variable_scope('branch_pool'):
                  branch_pool = ops.avg_pool(net, [3, 3])
                  branch_pool = ops.conv2d(branch_pool, 32, [1, 1])
                net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
                end_points['mixed_35x35x256a'] = net
              # mixed_1: 35 x 35 x 288.
              with tf.variable_scope('mixed_35x35x288a'):
                with tf.variable_scope('branch1x1'):
                  branch1x1 = ops.conv2d(net, 64, [1, 1])
                with tf.variable_scope('branch5x5'):
                  branch5x5 = ops.conv2d(net, 48, [1, 1])
                  branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                with tf.variable_scope('branch3x3dbl'):
                  branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                  branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                  branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                with tf.variable_scope('branch_pool'):
                  branch_pool = ops.avg_pool(net, [3, 3])
                  branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
                net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
                end_points['mixed_35x35x288a'] = net
              # mixed_2: 35 x 35 x 288.
              with tf.variable_scope('mixed_35x35x288b'):
                with tf.variable_scope('branch1x1'):
                  branch1x1 = ops.conv2d(net, 64, [1, 1])
                with tf.variable_scope('branch5x5'):
                  branch5x5 = ops.conv2d(net, 48, [1, 1])
                  branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
                with tf.variable_scope('branch3x3dbl'):
                  branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                  branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                  branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                with tf.variable_scope('branch_pool'):
                  branch_pool = ops.avg_pool(net, [3, 3])
                  branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
                net = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])
                end_points['mixed_35x35x288b'] = net
              # mixed_3: 17 x 17 x 768.
              with tf.variable_scope('mixed_17x17x768a'):
                with tf.variable_scope('branch3x3'):
                  branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
                with tf.variable_scope('branch3x3dbl'):
                  branch3x3dbl = ops.conv2d(net, 64, [1, 1])
                  branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
                  branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
                                            stride=2, padding='VALID')
                with tf.variable_scope('branch_pool'):
                  branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
                net = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])
                end_points['mixed_17x17x768a'] = net
              # mixed4: 17 x 17 x 768.
              with tf.variable_scope('mixed_17x17x768b'):
                with tf.variable_scope('branch1x1'):
                  branch1x1 = ops.conv2d(net, 192, [1, 1])
                with tf.variable_scope('branch7x7'):
                  branch7x7 = ops.conv2d(net, 128, [1, 1])
                  branch7x7 = ops.conv2d(branch7x7, 128, [1, 7])
                  branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
                with tf.variable_scope('branch7x7dbl'):
                  branch7x7dbl = ops.conv2d(net, 128, [1, 1])
                  branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
                  branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [1, 7])
                  branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
                  branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
                with tf.variable_scope('branch_pool'):
                  branch_pool = ops.avg_pool(net, [3, 3])
                  branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
                net = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])
                end_points['mixed_17x17x768b'] = net
          
              init = tf.global_variables_initializer()
              sess.run(init)
              pool_outs = sess.run(net)
      
t2 = datetime.datetime.now()

# previous part of the model runs here:
payload = json.dumps({'instances' : pool_outs.tolist()})
r = requests.post('http://104.196.229.77:8501/v1/models/partial_inception_v1_mixed1:predict', data=payload)
t3 = datetime.datetime.now()
print("Time on edge: " + str(t2 - t1))
print("Time roundtrip to cloud and back: " + str(t3 - t2))
print("Total time: " + str(t3 - t1))
print(r.text)

