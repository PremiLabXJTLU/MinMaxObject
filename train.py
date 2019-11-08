import model
import tensorflow as tf
import time,sys
import NIN_input as loader
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class MinMaxObject(object):
    def __init__(self,sess,flag):
        self.sess=sess
        self.step=tf.Variable(0, trainable=False,name='global_step')
        self.global_step=0
        self.flags=flag
        self.use_MinMax=True
        self.use_net='NIN'
        self.lamb=4e-6
        if self.use_net=='NIN':
            self.model = model.NIN
            self.momentum=0.95
            self.batch_size=128
        elif self.use_net=='ResNet':
            self.model = model.ResNet
            self.momentum=0.9
            self.batch_size=128
        elif self.use_net=='DenseNet':
            self.model = model.DenseNet
            self.momentum=0.9
            self.batch_size=64
        self.num_classes=10
        self.saved_dict = {}
        self.image_size=[32,32,3]
        self._build_model()


    def _build_model(self):
        self.is_train=tf.placeholder(tf.bool, None, name='training_phase')
        self.reset = tf.placeholder(tf.bool, None, name='reset_phase')
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.image=tf.placeholder(tf.float32,[None,self.image_size[0],self.image_size[1],self.image_size[2]],name='image')
        self.labels=tf.placeholder(tf.int64,[None,1],name='label')
        self.logits,self.embed=self.model(self.image,self.is_train,self.num_classes,False,name=self.use_net+'-cifar')
        self.pred_classes = tf.argmax(tf.nn.softmax(self.logits,name='softmax'), axis=1,name='pred_classes')
        self.valid_label=tf.reshape(self.labels,[-1],name='valid_label')
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.pred_classes, self.valid_label,name='equal'), tf.float32),name='accuracy')

        self.ce_loss= tf.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.valid_label)
        self.wd_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.minmax_loss,self.minmax_value=model.MinMaxObject(self.embed,self.labels,self.is_train,self.num_classes,self.batch_size,lamb=self.lamb,reset=self.reset)
        self.loss=self.ce_loss+self.wd_loss
        if self.use_MinMax:
            self.loss +=self.minmax_loss
        b=[]
        for var in tf.trainable_variables():
            if 'kernel' in var.name:
                c=1
                a=var.get_shape().as_list()
                for i in range(len(a)):
                    c*=a[i]
                print(var.name,var.shape,c)
                b.append(c)
                del a,c
        print(len(b),sum(b))


    def train(self):
        load_data = loader.load_data('cifar-10')
        test_images, test_labels = load_data.load(self.use_net)
        opt = tf.train.MomentumOptimizer(self.lr, self.momentum)
        grads_and_vars = opt.compute_gradients(self.loss, tf.trainable_variables())
        capped_grads_and_vars = []
        for gv in grads_and_vars:
            if 'mlpconv3/conv1x1_2/' in gv[1].name:
                capped_grads_and_vars.append((0.1 * gv[0], gv[1]))
            elif 'bias' in gv[1].name:
                capped_grads_and_vars.append((2 * gv[0], gv[1]))
            else:
                capped_grads_and_vars.append((gv[0], gv[1]))
        self.train = opt.apply_gradients(capped_grads_and_vars,global_step=self.step)
        var_list = []
        for var in tf.trainable_variables():
            var_list.append(var)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
        if (self.flags.is_continue_train):
            ckpt = tf.train.get_checkpoint_state("./parameters/")
            if ckpt and ckpt.model_checkpoint_path:
                print('successfully load')
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        best_acc=0
        cur_learning_rate=0.1
        count=0
        tmplist=[]
        for epoch in range(320):
            counter=np.array([0.,0.,0.,0.,0.,0.,0.,])
            test_counter=np.array([0.,0.])
            st=time.time()
            train_images, train_labels = load_data.shuffle()
            reset=True
            for i in range(int(50000/self.batch_size)):
                _,ce,wd,mml,ls,ac,mmv,self.global_step= self.sess.run([self.train,self.ce_loss,self.wd_loss,self.minmax_loss,self.loss,self.accuracy,self.minmax_value,self.step], feed_dict={self.image:train_images[i*self.batch_size:(i+1)*self.batch_size],self.labels:train_labels[i*self.batch_size:(i+1)*self.batch_size],self.lr: cur_learning_rate, self.is_train: True,self.reset:reset})
                counter+=np.array([ce,wd,mml,ls,ac,mmv[0],mmv[1]])
                reset=False
            if epoch in [1,10,60,120,160,200,240]:
                logistsp = np.empty(shape=[0, 10])
                for i in range(0,100,1):
                    logits,tls,tac=self.sess.run([self.logits,self.loss,self.accuracy],feed_dict={self.image:test_images[i*100:(i+1)*100],self.labels:test_labels[i*100:(i+1)*100],self.is_train:False,self.reset:reset})
                    test_counter+=np.array([tls,tac])

                    logistsp = np.vstack((logistsp, logits))



                tes=test_labels.reshape(-1)
                asd=np.argwhere(np.logical_or(tes==3,tes==5)).reshape(-1)
                slog=logistsp[asd]
                lbl=tes[asd]
                lbl=(lbl==5).astype(np.int)

                self.SNE(slog,lbl,epoch)
            counter /= int(50000 / 128)
            test_counter /= 100

            self.writer.add_summary(self.get_summary(counter, True), epoch)
            self.writer.add_summary(self.get_summary(test_counter, False), epoch)
            self.saved_dict[epoch]=[counter.tolist(),test_counter.tolist()]
            if best_acc<test_counter[1]:
                count=0
                best_acc=test_counter[1]
                if epoch > 140:
                    name='NIN_%d_%1.4f.ckpt'%(epoch,best_acc)
                    self.saver.save(self.sess, "./parameters/"+name, global_step=self.global_step)
            else:
                count+=1
            if epoch in [80,160,240,320]:
                cur_learning_rate *= 0.1


            sys.stdout.write('\r[ %d / 320 ][ %3.3fs ] train/test/best accuracy: %2.4f/%2.4f/%2.4f || ce/wd/mml loss: %2.4f/%2.4f/%2.4f ||  Inner-class/Inter-class:  %1.3f / %1.3f  || lr: %1.6f count:  %d' % (
             epoch, time.time() - st, counter[4], test_counter[1], best_acc, counter[0], counter[1], counter[2], counter[5], counter[6],cur_learning_rate,count))
            sys.stdout.flush()
            if len(tmplist)==3:
                print(tmplist)
                break
        self.save_pkl()
    def SNE(self,logistsp,labels,e):
        start_time=time.time()
        label_flat = labels.reshape(-1)
        #X_tsne = TSNE(n_components=2, learning_rate=400, init='pca', random_state=0).fit_transform(slog)  # tsne = TSNE(n_components=2, random_state=0)
        X_pca = PCA().fit_transform(logistsp)
        fig=plt.figure('t-sne', figsize=(16, 12))
        ax1=fig.add_subplot(111)
        ax1.set_title('Data Distribution')
        #    plt.subplot(121)
        # cm = plt.cm.get_cmap('PuBuGn'),cmap=cm
        x1=label_flat==0
        y1=X_pca[x1]
        x2=label_flat==1
        y2=X_pca[x2]
        ax1.scatter(y1[:, 0], y1[:, 1], c='r',marker='^', s=15, vmin=0, vmax=1)
        ax1.scatter(y2[:, 0], y2[:, 1], c='g',marker='v',s=15, vmin=0, vmax=1)
        # 设置X轴标签
        #plt.xlabel('X')

        #plt.ylabel('Y')

        t=time.time()
        picname = 't-sne-' + str(e) + '.png'
        plt.savefig(picname, dpi=300)
        plt.close()
        #print('TSNE cost time: %3.2f' % (time.time() - start_time))
        return

    def get_summary(self, counter, is_train):
        if is_train:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="ce_loss", simple_value=counter[0]),
                tf.Summary.Value(tag="wd_loss", simple_value=counter[1]),
                tf.Summary.Value(tag="loss/train_loss", simple_value=counter[3]),
                tf.Summary.Value(tag="accuracy/train_accuracy", simple_value=counter[4]),
            ])
            return summary
        else:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="loss/test_loss", simple_value=counter[0]),
                tf.Summary.Value(tag="accuracy/test_accuracy", simple_value=counter[1]),
            ])
            return summary

    def save_pkl(self):
        import pickle
        with open('data.pkl', 'wb') as f:
            pickle.dump(self.saved_dict, f)




