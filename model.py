from ops import mlpconv, conv, batch_norm_layer, residual_block, dense_block, transition_layer
import tensorflow as tf
import numpy as np


def NIN(image, is_train, num_classes=10, reuse=False, name='NIN-cifar'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        mlp1 = mlpconv(image, 5, [192, 160, 96], True, is_train, 'mlpconv1')
        mlp2 = mlpconv(mlp1, 5, [192, 192, 192], True, is_train, 'mlpconv2')
        mlp3 = mlpconv(mlp2, 3, [192, 192, num_classes], False, is_train, 'mlpconv3')
        ave_vec = tf.nn.avg_pool(mlp3, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID')
        logits = tf.reshape(ave_vec, [-1, num_classes])
        embed = logits
    return logits, embed


def ResNet(image, is_train, num_classes=10, reuse=False, name='ResNet-cifar'):
    channel_list = [16, 32, 64]
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        Res_Block_0 = tf.nn.relu(batch_norm_layer(conv(image, 16, 3, 'Block_0/res_conv_0', s=1, use_bias=False), is_train))
        Res_Block_1 = residual_block(Res_Block_0, channel_list[0], False, is_train, 'Block_1')
        Res_Block_2 = residual_block(Res_Block_1, channel_list[1], True, is_train, 'Block_2')
        Res_Block_3 = residual_block(Res_Block_2, channel_list[2], True, is_train, 'Block_3')
        ave_vec = tf.reshape(tf.nn.avg_pool(Res_Block_3, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID'), [-1, channel_list[2]])
        logits = tf.layers.dense(ave_vec, num_classes, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='logits')
        embed = logits
    return logits, embed


def DenseNet(image, is_train, num_classes=10, reuse=False, name='DenseNet-cifar'):
    L = 100
    K = 12
    C = 0.5
    keep_prob = 0.8
    nb_layers = int(((L - 4) / 3) / 2)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        Dense_Block_0 = conv(tf.nn.relu(batch_norm_layer(image, is_train)), 2 * K, 3, 'Block_0_conv0', use_bias=False)
        Dense_Block_1 = dense_block(Dense_Block_0, nb_layers, is_train, 'Block_1', growthRate=K, dropout_rate=keep_prob)
        Transition_Layer_1 = transition_layer(Dense_Block_1, is_train, 'Block_1_transition', compression=C, dropout_rate=keep_prob)
        Dense_Block_2 = dense_block(Transition_Layer_1, nb_layers, is_train, 'Block_2', growthRate=K, dropout_rate=keep_prob)
        Transition_Layer_2 = transition_layer(Dense_Block_2, is_train, 'Block_2_transition', compression=C, dropout_rate=keep_prob)
        Dense_Block_3 = dense_block(Transition_Layer_2, nb_layers, is_train, 'Block_3', growthRate=K, dropout_rate=keep_prob)
        x = tf.nn.relu(batch_norm_layer(Dense_Block_3, is_train))
        ave_vec = tf.reshape(tf.nn.avg_pool(x, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID'), [-1, x.get_shape().as_list()[-1]])
        logits = tf.layers.dense(ave_vec, num_classes, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='logits')
        embed = logits
    return logits, embed

def MinMaxObject(embed,labels,is_train,num_classes,batch_size=128,lamb=1e-5,reset=False):
    def get_Adjacency_Matrix(D,L,batch_size,num_classes,k1=5,k2=10):
        labels_flat = np.tile(L.reshape(1, batch_size), (batch_size, 1))
        Gi = np.equal(labels_flat, labels_flat.T).astype(np.float32) - np.diag(np.ones((batch_size)))
        AGi = D * Gi
        top_k1 = np.argsort(np.where(AGi == 0, np.max(AGi, axis=1, keepdims=True), AGi), axis=1)
        top_k1_value = np.array([AGi[i, top_k1[i, k1]] for i in range(batch_size)]).reshape((batch_size, 1))
        AGi = np.where(np.logical_and(AGi < top_k1_value, AGi > 0), 1, 0)
        AGi = np.logical_or(AGi, AGi.T).astype(np.float32)
        IGi = np.invert(np.equal(labels_flat, labels_flat.T)).astype(np.float32)
        Gp = np.zeros((num_classes, batch_size, batch_size), dtype=np.float32)
        for i in range(batch_size):
            Gp[np.int(L[i]), i] = IGi[i]
        AGp = np.reshape(D * Gp, [num_classes, batch_size * batch_size])
        top_k2 = np.argsort(np.where(AGp == 0, np.max(AGp, axis=1, keepdims=True), AGp), axis=1)
        top_k2_value = np.array([AGp[j, top_k2[j, k2]] for j in range(num_classes)]).reshape((num_classes, 1))
        AGp = np.where(np.logical_and(AGp < top_k2_value, AGp > 0), 1, 0)
        AGp = np.reshape(np.sum(AGp, axis=0), [batch_size, batch_size])
        AGp = np.logical_or(AGp, AGp.T).astype(np.float32)
        recorder=[np.sum(D*AGi),np.sum(D*AGp),np.sum(AGi),np.sum(AGp)]
        return (AGi-AGp).astype(np.float32),recorder
    batch_size = tf.where(is_train,batch_size,100)
    shape=embed.get_shape().as_list()
    if len(shape)==2:
        embed_size=shape[1]
    else:
        embed_size=shape[1]*shape[2]*shape[3]
    embedding_layer = tf.reshape(embed, [batch_size, embed_size])
    a = tf.reshape(embedding_layer, [batch_size*embed_size])
    a = tf.reshape(tf.tile(a, [batch_size]), [batch_size, batch_size, embed_size])
    b = tf.transpose(a, [1, 0, 2])
    Distance_Matrix = tf.reduce_sum(tf.square(a - b), axis=2)
    [Adjacency_Matrix,MinMaxRecoder]=tf.py_func(get_Adjacency_Matrix,[Distance_Matrix,labels,batch_size,num_classes],[tf.float32,tf.float32])
    if True:
        return tf.reduce_sum(Distance_Matrix * Adjacency_Matrix) * lamb,MinMaxRecoder
    elif True:
        Laplacian_Matrix=tf.diag(tf.reshape(tf.reduce_sum(Adjacency_Matrix,axis=1),shape=[-1]))-Adjacency_Matrix
        return 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(embedding_layer, [1, 0]), Laplacian_Matrix), embedding_layer)) * lamb,MinMaxRecoder
    else:
        pass


def MinMaxObject2(embed,labels,is_train,num_classes,batch_size,lamb,reset,avg_pool=False,laplacian=False):
    def get_mc(label,batch_size,i):
        label=label.flatten()
        tmpindex = np.arange(0, batch_size, 1)
        output=tmpindex[label==i]
        if  output.tolist()==[]:
            return output,True
        else:
            return output,False
    embedding_layer=embed
    batch_size = tf.where(is_train,batch_size,100)
    shape = embedding_layer.get_shape().as_list()
    if len(shape) == 2:
        embed_size = shape[1]
    else:
        embed_size = shape[1] * shape[2] * shape[3]
    embedding_layer = tf.reshape(embedding_layer, [batch_size, embed_size])
    mc=[]
    nc=[]
    for i in range(num_classes):
        index, is_empty = tf.py_func(get_mc, [labels, batch_size,i], [tf.int32,tf.bool])
        mean=tf.cond(is_empty,lambda:tf.zeros([embed_size]),lambda:tf.reduce_mean(tf.gather(embedding_layer,index),axis=0))#tf.dynamic_partition(embedding_layer,index,self.batch_size)tf.gather(embedding_layer,index)
        mc.append(mean)
        nc.append(tf.cast(tf.shape(index),dtype=tf.float32))
    mc=tf.reshape(tf.stack(mc,axis=0),[num_classes,embed_size])
    nc=tf.reshape(tf.stack(nc,axis=0),[num_classes,1])
    Sw=tf.reduce_sum(tf.square(embedding_layer-tf.gather(mc,labels)))
    m=tf.reduce_mean(mc,axis=0,keep_dims=True)
    Sb=tf.reduce_sum(nc*tf.square((mc-m)))
    return -lamb*Sb/Sw,[(Sw/tf.cast(batch_size,tf.float32))**0.5,(Sb/tf.cast(batch_size,tf.float32))**0.5,mc,nc]

'''
    mc=tf.reshape(tf.stack(mc,axis=0),[num_classes,embed_size])
    nc=tf.reshape(tf.stack(nc,axis=0),[num_classes,1])
    Sw=tf.reduce_sum(tf.square(embedding_layer-tf.gather(mc,labels)))
    m=tf.reduce_mean(mc,axis=0,keep_dims=True)
    Sb=tf.reduce_sum(nc*tf.square((mc-m)))
    return -lamb*Sb/Sw,[(Sw/tf.cast(batch_size,tf.float32))**0.5,(Sb/tf.cast(batch_size,tf.float32))**0.5,mc,nc]
    
    mc = tf.reshape(tf.stack(mc, axis=0), [num_classes, embed_size])
    nc = tf.reshape(tf.stack(nc, axis=0), [num_classes, 1])

    tmp_mc=tf.Variable(tf.zeros_like(mc),trainable=False)
    tmp_nc=tf.Variable(tf.zeros_like(nc),trainable=False)
    up1=tf.cond(reset,lambda:tf.assign(tmp_mc,mc),lambda:tf.assign_add(tmp_mc,mc))
    up2=tf.cond(reset,lambda:tf.assign(tmp_nc,nc),lambda:tf.assign_add(tmp_nc,nc))
    #update=tf.group(update_mc,update_nc)
    with tf.control_dependencies([up1,up2]):
        new_mc=(mc*nc+tmp_mc*tmp_nc)/(nc+tmp_nc)
        Sw=tf.reduce_sum(tf.square(embedding_layer-tf.gather(new_mc,labels)))
        m=tf.reduce_mean(new_mc,axis=0,keep_dims=True)
        Sb=tf.reduce_sum(nc*tf.square((new_mc-m)))
        return -lamb*Sb/Sw,[(Sw/tf.cast(batch_size,tf.float32))**0.5,(Sb/tf.cast(batch_size,tf.float32))**0.5,m,nc+tmp_nc]#20 50w
'''