import tensorflow as tf

def batch_norm_layer(x, train_phase):
    with tf.variable_scope('BN'):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)

        axises = list(range(len(x.get_shape() ) - 1))
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv(x,out_channel,ks,name,s=1,use_bias=True):
    with tf.variable_scope(name):
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.05)
        bias_initializer = tf.constant_initializer(0)
        regularizer = tf.contrib.layers.l2_regularizer(0.0001)
        return tf.layers.conv2d(x, out_channel, ks,s, padding='SAME',use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=regularizer,activation=tf.nn.relu)

def mlpconv(x, ks, channel_list, max_pooling, is_train, name):
    with tf.variable_scope(name):
        h = conv(x, channel_list[0], ks, name='conv3x3_0')
        h = conv(h, channel_list[1], 1, name='conv1x1_1')
        h = conv(h, channel_list[2], 1, name='conv1x1_2')
        if max_pooling:
            h = tf.nn.max_pool(h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool')
            h = tf.cond(is_train, lambda: tf.nn.dropout(h, 0.5), lambda:  tf.nn.dropout(h, 1), name='drop_out')
        return h


def residual_block(x,channel_number,is_downsample,is_train,name,n=3):
    with tf.variable_scope(name):
        if is_downsample:
            h= tf.nn.relu(batch_norm_layer(conv(x,channel_number,3,'res_conv_0',s=2,use_bias=False),is_train))
            x=tf.nn.avg_pool(x, ksize=[1, 1, 1, 1], strides=[1, 2, 2, 1], padding='SAME')
            x= tf.concat([x, tf.zeros_like(x, dtype=tf.float32)], axis=3)#conv(inputs, 1, output_number, is_train, s=2)
        else:
            h = tf.nn.relu(batch_norm_layer(conv(x, channel_number, 3, 'res_conv_0', s=1, use_bias=False),is_train))
        x=tf.nn.relu(batch_norm_layer(conv(h,channel_number,3,'res_conv_1',use_bias=False),is_train)+x)
        for i in range(n-1):
            h=tf.nn.relu(batch_norm_layer(conv(x, channel_number, 3, 'res_conv_'+str(i*2+2), s=1, use_bias=False),is_train))
            x=tf.nn.relu(batch_norm_layer(conv(h,channel_number,3,'res_conv_'+str(i*2+3),use_bias=False),is_train)+x)
    return x

def bottleneck_layer(x,is_train, name,growthRate=12,dropout_rate=0.9):
    #DenseNet-BC
    with tf.variable_scope(name):
        x=conv(tf.nn.relu(batch_norm_layer(x,is_train)),4*growthRate,1,name+ '_conv1',use_bias=False)
        x=tf.cond(is_train, lambda: tf.nn.dropout(x, dropout_rate), lambda: x)
        x=conv(tf.nn.relu(batch_norm_layer(x,is_train)),growthRate,3,name+ '_conv2',use_bias=False)
        x=tf.cond(is_train, lambda: tf.nn.dropout(x, dropout_rate), lambda: x)
        return x
def transition_layer( x,is_train, name,compression=0.5,dropout_rate=0.9):
    with tf.variable_scope(name):
        x = conv(tf.nn.relu(batch_norm_layer(x, is_train)), int(x.get_shape().as_list()[-1]*compression), 1, name + '_conv1', use_bias=False)
        x = tf.cond(is_train, lambda: tf.nn.dropout(x, dropout_rate), lambda: x)
        x = tf.nn.avg_pool(x, [1,2,2,1],[1,2,2,1],padding='VALID')
        return x
def dense_block(input_x, nb_layers,is_train, name,growthRate=12,dropout_rate=0.9):
    with tf.variable_scope(name):
        layers_concat = list()
        layers_concat.append(input_x)
        x = bottleneck_layer(input_x, is_train,name + '_bottleN_' + str(0),growthRate=growthRate,dropout_rate=dropout_rate)
        layers_concat.append(x)
        for i in range(nb_layers - 1):
            x = tf.concat(layers_concat,axis=-1)
            x = bottleneck_layer(x, is_train,name + '_bottleN_' + str(i + 1),growthRate=growthRate,dropout_rate=dropout_rate)
            layers_concat.append(x)
        x=tf.concat(layers_concat,axis=-1)
        return x
