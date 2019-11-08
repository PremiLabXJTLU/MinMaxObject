import numpy as np
import tensorflow as tf
import os
# loader.encode_to_tfrecords()
# train_image,train_label=loader.decode_from_tfrecords('E:/cifar10_train_zca.tfrecord',is_shuffle=True,is_train=False,batch_size=128)
# test_image,test_label=loader.decode_from_tfrecords('E:/cifar10_test_zca.tfrecord',is_shuffle=False,is_train=False,batch_size=100)
# self.image =tf.cond(self.is_train,lambda :train_image,lambda :test_image)
# self.labels =tf.cond(self.is_train,lambda :train_label,lambda :test_label)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def zca_whitening(image,test):  # n*m, m parament number, n example number
    _mean = np.mean(image, axis=0)
    _std = np.mean(image, axis=0)
    image = (image - _mean) /_std
    test=(test-_mean)/_std
    sigma = np.dot(image.T, image) / (image.T.shape[1] - 1)  # m*m
    eig_val, eig_vec = np.linalg.eig(sigma)
    S_sqrt = np.sqrt(np.diag(eig_val))
    ZCAMatrix = np.dot(eig_vec, np.dot(np.linalg.inv(S_sqrt), eig_vec.T))
    image = np.dot(ZCAMatrix, image.T)
    image = image.T.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test = np.dot(ZCAMatrix, test.T)
    test = test.T.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return image,test

def encode_to_tfrecords(train_filename='cifar10_train_zca.tfrecord',test_filename='cifar10_test_zca.tfrecord'):
    if os.path.isfile('E:/'+train_filename) and os.path.isfile('E:/'+test_filename):
        print('tfrecord ojbk')
    else:

        path_list = ["E:/cifar-10/data_batch_"+str(i+1) for i in range(5)]
        first=True
        image=None
        label=None
        for path in path_list:
            dict=unpickle(path)
            if first:
                image=dict[b'data'].astype(np.float32)
                label=np.array(dict[b'labels'], dtype=np.int64).reshape(-1,1)
                first=False
            else:
                img=dict[b'data'].astype(np.float32)
                image=np.vstack((image,img))
                lab = np.array(dict[b'labels'], dtype=np.int64).reshape(-1,1)
                label = np.vstack((label,lab))
        test_path='E:/cifar-10/test_batch'
        dict=unpickle(test_path)
        test=dict[b'data'].astype(np.float32)
        test_label=np.array(dict[b'labels'], dtype=np.int64).reshape(-1,1)
        image,test=zca_whitening(image,test)
        print(np.shape(image),np.shape(label),np.shape(test),np.shape(test_label))

        writer = tf.python_io.TFRecordWriter('E:/' + train_filename)
        for i in range(50000):
            if i %1000==0:
                print(i)
            x_raw = image[i].tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label[i]))
                }))
            writer.write(example.SerializeToString())
        writer.close()
        writer = tf.python_io.TFRecordWriter('E:/' + test_filename)
        for i in range(10000):
            if i %1000==0:
                print(i)
            x_raw = test[i].tostring()
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_raw])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=test_label[i]))
                }))
            writer.write(example.SerializeToString())
        writer.close()


def decode_from_tfrecords(tfrecords_filename,is_shuffle,is_train,batch_size):
    num_threads=16
    min_after_dequeue = 20000
    capacity = min_after_dequeue + 3 * batch_size

    reader = tf.TFRecordReader()
    filename_queue=tf.train.string_input_producer([tfrecords_filename])
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image':tf.FixedLenFeature([], tf.string),
                                           'label':tf.FixedLenFeature([1], tf.int64)
                                       })

    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, [32,32,3])
    label= features['label']
    label = tf.reshape(label, [1])
    if is_train:
        image=tf.image.resize_image_with_crop_or_pad(image,target_height=36, target_width=36)
        image=tf.random_crop(image,[32,32,3])
        image=tf.image.random_flip_left_right(image)
    if is_shuffle:
        image, label = tf.train.shuffle_batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=num_threads,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    else:
        image, label = tf.train.batch([image, label],
                                      batch_size=batch_size,
                                      num_threads=num_threads,
                                      capacity=capacity)

    return image, label
