import tensorflow as tf

flags = tf.app.flags

#    hyper parameters      
flags.DEFINE_float('clip_min', 1e-8, 'epsilon')
flags.DEFINE_float('clip_max', 1000, 'clip_max')

flags.DEFINE_integer('max_count', 100, 'max data m')
flags.DEFINE_integer('epoch', 30, 'epoch')
flags.DEFINE_integer('batch_size', 20, 'batch size')
flags.DEFINE_integer('iter_routing', 1, 'number of iterations')

#    structure parameters
flags.DEFINE_integer('A', 64, 'number of channels in output from ReLU Conv1')
flags.DEFINE_integer('B', 8, 'number of channels in output from PrimaryCaps')
flags.DEFINE_integer('C', 16, 'number of channels in output from ConvCaps1')
flags.DEFINE_integer('D', 16, 'number of channels in output from ConvCaps2')


#   environment setting

flags.DEFINE_string('dataset', './data', 'the path for dataset')
cfg = tf.app.flags.FLAGS