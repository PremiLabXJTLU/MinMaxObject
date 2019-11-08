import tensorflow as tf
import time,os
from train import MinMaxObject
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epoch', 200, "total epoch")
tf.app.flags.DEFINE_boolean('is_continue_train',False,'')


def main(_):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth=True

    tf_config.gpu_options.per_process_gpu_memory_fraction = 1

    with tf.Session(config=tf_config) as sess:
        model=MinMaxObject(sess,FLAGS)
        model.train()


if __name__ == '__main__':
    tf.app.run()

