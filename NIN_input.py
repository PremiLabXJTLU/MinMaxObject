import numpy as np
import time
class load_data():
    def __init__(self,cifar):
        self.cifar = cifar
        self.train_image = []

        self.train_label_flat = []
        self.test_image = []

        self.test_label_flat = []
        self.data_augmentation=False

    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def dense_to_one_hot(self,labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def preprocess(self,net='NIN'):  # n*m, m parament number, n example number
        if net=='NIN':
            print('ZCA Whitning')
            _mean = np.mean(self.train_image, axis=0)
            _std = np.mean(self.train_image, axis=0)
            self.train_image = (self.train_image - _mean) /_std
            self.test_image=(self.test_image-_mean)/_std
            sigma = np.dot(self.train_image.T, self.train_image) / (self.train_image.T.shape[1] - 1)  # m*m
            eig_val, eig_vec = np.linalg.eig(sigma)
            S_sqrt = np.sqrt(np.diag(eig_val))
            ZCAMatrix = np.dot(eig_vec, np.dot(np.linalg.inv(S_sqrt), eig_vec.T))
            self.train_image = np.dot(ZCAMatrix, self.train_image.T)
            self.train_image = self.train_image.T.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.test_image = np.dot(ZCAMatrix, self.test_image.T)
            self.test_image = self.test_image.T.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        elif net=='ResNet':
            # per-pixel mean subtracted
            print('per-pixel mean subtracted')
            self.train_image = self.train_image.reshape(-1, 3, 32*32)
            self.test_image = self.test_image.reshape(-1, 3, 32*32)
            _mean = np.mean(self.train_image, axis=0)
            _mean = np.mean(_mean, axis=0)
            self.train_image = (self.train_image - _mean)
            self.test_image = (self.test_image - _mean)
            self.train_image = self.train_image.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.test_image = self.test_image.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            self.data_augmentation = True
        elif net=='DenseNet':
            print('normalize the data using the channel means and standard deviations')
            self.train_image = self.train_image.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(-1,3)
            self.test_image = self.test_image.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).reshape(-1,3)
            _mean = np.mean(self.train_image, axis=0)
            _std = np.std(self.train_image, axis=0,ddof=1)
            self.train_image = (self.train_image - _mean)/_std
            self.test_image = (self.test_image - _mean)/_std
            self.train_image = self.train_image.reshape(-1, 32, 32,3)
            self.test_image = self.test_image.reshape(-1, 32, 32,3)
            self.data_augmentation = True
    def load(self,net='NIN'):
        if self.cifar == 'cifar-10':
            print('Begin to load cifar-10')
            start_time=time.time()
            path = "G:\PYTHON\cifar-10\data_batch_"
            path1 = path + str(1)
            dict1 = self.unpickle(path1)
            self.train_image = dict1[b'data'].astype(np.float32)
            self.train_label_flat = np.array(dict1[b'labels'], dtype=np.int64).reshape(-1, 1)
            for i in range(4):
                paths = path + str(i + 2)
                dicts = self.unpickle(paths)
                datas = dicts[b'data'].astype(np.float32)
                self.train_image = np.vstack((self.train_image, datas))
                labels = np.array(dicts[b'labels'], dtype=np.int64).reshape(-1, 1)
                self.train_label_flat = np.vstack((self.train_label_flat, labels))

            path = r'G:\PYTHON\cifar-10\test_batch'
            dict1 = self.unpickle(path)
            self.test_label_flat = np.array(dict1[b'labels'], dtype=np.int64).reshape(-1, 1)
            self.test_image = dict1[b'data'].astype(np.float32)
            self.preprocess(net)
            print('Load Cifar-10 successfully with time %d'%(time.time()-start_time))
            return self.test_image, self.test_label_flat
        elif self.cifar == 'cifar-100':
            print('Begin to load cifar-100')
            start_time=time.time()
            path = r'G:\PYTHON\cifar-100\train'
            dict = self.unpickle(path)
            self.train_label_flat = np.array(dict[b'fine_labels'], dtype=np.int64).reshape(-1, 1)
            self.train_image = dict[b'data'].astype(np.float32) # 500000 3072


            path = r'G:\PYTHON\cifar-100\test'
            dict = self.unpickle(path)
            self.test_label_flat = np.array(dict[b'fine_labels'], dtype=np.int64).reshape(-1, 1)
            self.test_image = dict[b'data']  # 500000 3072

            self.preprocess(net)
            print('Load Cifar-100 successfully with time %d'%(time.time()-start_time))
            return self.test_image, self.test_label_flat
        else:
            print('ERROR: The input must be named as "cifar-10" or "cifar-100" ')
            return  self.test_image,  self.test_label_flat

    def shuffle(self):
        index = np.arange(50000)
        np.random.shuffle(index)
        if self.data_augmentation:
            return self.augmentation(self.train_image[index,:,:,:]), self.train_label_flat[index,:]
        else:
            return self.train_image[index, :, :, :], self.train_label_flat[index, :]

    def augmentation(self,input):
        output=np.zeros((50000,32,32,3))
        i=0
        for image in input:
            image=np.pad(image, ((2, 2), (2, 2),(0,0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            image = self.crop_picture_randomly(image, sizes=(32, 32))
            image = self.horizontal_flipping(image, 0.5)
            output[i]=image
            i+=1
            #output = np.append(output, np.expand_dims(image,axis=0), axis=0)
        return output

    def crop_picture_randomly(self,x_or_probability, sizes=(32, 32)):
        y, x, channel = x_or_probability.shape
        keypoints = self.get_keypoints_randomly_for_cropping((y, x, channel), sizes)
        start_y, end_y = keypoints[0]
        start_x, end_x = keypoints[1]
        return x_or_probability[start_y:end_y, start_x:end_x]

    def get_keypoints_randomly_for_cropping(self,picture_shape, sizes):
        y, x, channel = picture_shape
        length_y, length_x = sizes
        # pick random number
        keypoint_y = self.pick_random_permutation(1, y - length_y + 1)[0]
        keypoint_x = self.pick_random_permutation(1, x - length_x + 1)[0]
        start_y = keypoint_y
        # end_y does not exceed picture_shape, because number is sampled from y - length_y + 1
        end_y = keypoint_y + length_y
        start_x = keypoint_x
        # end_x does not exceed picture_shape, because number is sampled from x - length_x + 1
        end_x = keypoint_x + length_x
        return ((start_y, end_y), (start_x, end_x))

    def pick_random_permutation(self,pick_number, sample_number, sort=False):
        pick_number = int(pick_number)
        sample_number = int(sample_number)
        sort = bool(sort)
        if sort:
            return np.sort(np.random.permutation(sample_number)[:pick_number])
        else:
            return np.random.permutation(sample_number)[:pick_number]
    def horizontal_flipping(self,x_or_probability, probability):
        import random
        if (random.random() > probability):
            output = np.fliplr(x_or_probability)
        else:
            output = x_or_probability
        return output





