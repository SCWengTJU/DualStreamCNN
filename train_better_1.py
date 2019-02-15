import tensorflow as tf
import numpy as np
import os

def getNameList():
    list=[]
    for i in range(1, 150):
        for j in range(1, 50):
            name = 'a' + str(i) + '_r' + str(j)
            if os.path.exists(tfrecords_dir + name +'.npy.tfrecord'):
                list.append(tfrecords_dir + name+'.npy.tfrecord')
    return list
pass

def read_and_decode(name_list):
    filename_queue = tf.train.string_input_producer(name_list, shuffle = True)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['img_raw'], tf.float64)
    image = tf.reshape(image, [39*67*67])
    image = tf.cast(image, tf.float32)
    label = tf.decode_raw(features['label'], tf.float64)#
    label = tf.reshape(label, [140])
    label = tf.cast(label, tf.float32)
    print type(label)
    return image,label, key
pass


def lrelu(input, alpha = 0.2):
	return (0.5 * (1 + alpha)) * input + (0.5 * (1 - alpha)) * tf.abs(input)

def weight_variable(shape, name):
    return tf.get_variable(initializer=tf.truncated_normal(shape, stddev=0.1), name=name, trainable=True)
pass

def bias_variable(shape, name):
    return tf.get_variable(initializer=tf.constant(0.1, shape=shape), name=name, trainable=True)
pass

def batch_normalization(x, n_out, phase_train):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    pass

    mean, var = tf.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
pass

def conv(x, conv_high, conv_width, in_shape, out_shape, name, stride_high, stride_width, padding, reuse):
    with tf.variable_scope('trunk') as scope:
        if reuse:
            scope.reuse_variables()
        x = tf.nn.conv2d(x, weight_variable([conv_high, conv_width, in_shape, out_shape], name), strides=[1, stride_high, stride_width, 1], padding=padding)
        x = tf.nn.bias_add(x, bias_variable([out_shape], name + '_bias'))
    return x
pass

def max_pool(x, pool_high, pool_width, name, stride_high, stride_width, padding, reuse):
    with tf.variable_scope('trunk') as scope:
        if reuse:
            scope.reuse_variables()
        x = tf.nn.max_pool(x, ksize=[1, pool_high, pool_width, 1], strides=[1, stride_high, stride_width, 1], padding='SAME')  # [10,16,100]
    return x
pass

def network(x, reuse, Bn_Choice):
    long_input = conv(x, 7, 7, 67, 100, 'long_7x7', 2, 2, 'SAME', reuse)
    long_input = batch_normalization(long_input, 100, phase_train= Bn_Choice )
    long_input = lrelu(long_input)

    mid_input = max_pool(long_input, 3, 3, 'mid_maxpool', 2, 2, 'SAME',reuse)
    mid_input = conv(mid_input, 1, 1, 100, 50, 'mid_1x1', 1, 1, 'SAME', reuse)
    mid_input = batch_normalization(mid_input, 50 , phase_train= Bn_Choice)
    mid_input = conv(mid_input, 3, 3, 50, 100, 'mid_3x3', 1, 1, 'SAME', reuse)
    mid_input = batch_normalization(mid_input, 100 , phase_train= Bn_Choice)

    shrot_input = max_pool(mid_input, 3, 3, 'short_maxpool', 2, 2, 'SAME', reuse)
        #long_1x1
    long_branch_input = conv(long_input, 1, 1, 100, 12, 'long_branch_1x1', 1, 1, 'SAME', reuse)
    long_branch_input = batch_normalization(long_branch_input, 12 , phase_train= Bn_Choice)
    long_branch_input = lrelu(long_branch_input)
    long_branch_input_gate_weight = tf.nn.sigmoid(long_branch_input[:, :, :, 6:])
    long_branch_input = long_branch_input[:, :, :, 0:6]
    long_branch_input = tf.multiply(long_branch_input, long_branch_input_gate_weight)
        #mid_1x1
    mid_branch_input = conv(mid_input, 1, 1, 100, 50, 'mid_branch_1x1', 1, 1, 'SAME', reuse)
    mid_branch_input = batch_normalization(mid_branch_input, 50 , phase_train= Bn_Choice)
    mid_branch_input = lrelu(mid_branch_input)
    mid_branch_input_gate_weight = tf.nn.sigmoid(mid_branch_input[:, :, :, 25:])
    mid_branch_input = mid_branch_input[:, :, :, 0:25]
    mid_branch_input = tf.multiply(mid_branch_input, mid_branch_input_gate_weight)
        # short_Conv 1x1+S1
    inception_1x1_S1 = conv(shrot_input, 1, 1, 100, 100, 'inception_first_1x1', 1, 1, 'SAME', reuse)
    inception_1x1_S1 = batch_normalization(inception_1x1_S1, 100 , phase_train= Bn_Choice)
    inception_1x1_S1 = lrelu(inception_1x1_S1)
    inception_1x1_S1_gate_weight = tf.nn.sigmoid(inception_1x1_S1[:, :, :, 50:])
    inception_1x1_S1 = inception_1x1_S1[:, :, :, 0:50]
    inception_1x1_S1 = tf.multiply(inception_1x1_S1, inception_1x1_S1_gate_weight)
        # short_Conv 3x3+S1
    inception_3x3_S1_reduce = conv(shrot_input, 1, 1, 100, 50, 'inception_second_1x1', 1, 1, 'SAME', reuse)
    inception_3x3_S1_reduce = batch_normalization(inception_3x3_S1_reduce,  50, phase_train= Bn_Choice)
    inception_3x3_S1_reduce = lrelu(inception_3x3_S1_reduce)
    inception_3x3_S1 = conv(inception_3x3_S1_reduce, 3, 3, 50, 200, 'inception_second_3x3', 1, 1, 'SAME', reuse)
    inception_3x3_S1 = batch_normalization(inception_3x3_S1,  200 , phase_train= Bn_Choice)
    inception_3x3_S1 = lrelu(inception_3x3_S1)
    inception_3x3_S1_gate_weight = tf.nn.sigmoid(inception_3x3_S1[:, :, :, 100:])
    inception_3x3_S1 = inception_3x3_S1[:, :, :, 0:100]
    inception_3x3_S1 = tf.multiply(inception_3x3_S1, inception_3x3_S1_gate_weight)
        # short_Conv 5x5+S1
    inception_5x5_S1_reduce = conv(shrot_input, 1, 1, 100, 50, 'inception_third_1x1', 1, 1, 'SAME', reuse)
    inception_5x5_S1_reduce = batch_normalization(inception_5x5_S1_reduce, 50, phase_train= Bn_Choice)
    inception_5x5_S1_reduce = lrelu(inception_5x5_S1_reduce)
    inception_5x5_S1 = conv(inception_5x5_S1_reduce, 5, 5, 50, 200, 'inception_third_5x5', 1, 1, 'SAME', reuse)
    inception_5x5_S1 = batch_normalization(inception_5x5_S1, 200 , phase_train= Bn_Choice)
    inception_5x5_S1 = lrelu(inception_5x5_S1)
    inception_5x5_S1_gate_weight = tf.nn.sigmoid(inception_5x5_S1[:, :, :, 100:])
    inception_5x5_S1 = inception_5x5_S1[:, :, :, 0:100]
    inception_5x5_S1 = tf.multiply(inception_5x5_S1, inception_5x5_S1_gate_weight)
        # short_MaxPool
    inception_MaxPool = max_pool(shrot_input, 3, 3, 'MaxPool_pool', 1, 1, 'SAME', reuse)
    inception_MaxPool = conv(inception_MaxPool, 1, 1, 100, 100, 'inception_fourth_1x1', 1, 1, 'SAME', reuse)
    inception_MaxPool = batch_normalization(inception_MaxPool, 100 , phase_train= Bn_Choice)
    inception_MaxPool = lrelu(inception_MaxPool)
    inception_MaxPool_gate_weight = tf.nn.sigmoid(inception_MaxPool[:, :, :, 50:])
    inception_MaxPool = inception_MaxPool[:, :, :, 0:50]
    inception_MaxPool = tf.multiply(inception_MaxPool, inception_MaxPool_gate_weight)

    long_branch_input = tf.contrib.layers.flatten(long_branch_input)
    mid_branch_input = tf.contrib.layers.flatten(mid_branch_input)
    inception_1x1_S1 = tf.contrib.layers.flatten(inception_1x1_S1)
    inception_3x3_S1 = tf.contrib.layers.flatten(inception_3x3_S1)
    inception_5x5_S1 = tf.contrib.layers.flatten(inception_5x5_S1)
    inception_MaxPool = tf.contrib.layers.flatten(inception_MaxPool)
    return tf.concat(axis=1, values=[long_branch_input, mid_branch_input, inception_1x1_S1, inception_3x3_S1, inception_5x5_S1, inception_MaxPool])
pass

def full_con_1(z):
    print z
    print z
    print z
    w_fc1 = weight_variable([43660, 500], 'full_con_1')
    b_fc1 = bias_variable([500], 'full_con_1_bias')
    h_pool2_flat = tf.reshape(z, [-1,43660])
    h_fc1 = lrelu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    return h_fc1
pass

def full_con_2(z):
    w_fc2 = weight_variable([500, 140], 'full_con_2')
    b_fc2 = bias_variable([140], 'full_con_2_bias')
    y_conv = tf.nn.softmax(tf.matmul(z, w_fc2) + b_fc2)
    return y_conv
pass

def interface(x):
    phase_train = tf.constant(True, dtype=tf.bool)
    another = x
    another =  tf.transpose(another, [0, 1, 3, 2])
    x = network(x, False, phase_train)
    another = network(another, True, phase_train)
    z = tf.concat(axis=1, values=[x, another])
    z = full_con_1(z)
    z = tf.nn.dropout(z, keep_prob=0.08)
    return full_con_2(z)
pass
#########################################Store the Input Data#################################
save_model_dir = './model_save/'
read_model_dir='./init_weight/'
tfrecords_dir = './train_67_tfrecord/'

if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

# #
sess = tf.InteractiveSession()
# #
fetches = {}
fetches_test = {}

name_list = getNameList()
length = len(name_list)
print 'done', length
#########################################################################################

learning_rate = 0.001
max_step = 10000



img, label, key = read_and_decode(name_list)
img_batch, label_batch, key_batch = tf.train.shuffle_batch([img, label, key], batch_size=1200, capacity=12000, min_after_dequeue=4000)
#

with tf.name_scope('input_shape'):
    y = label_batch
    xx1 = tf.reshape(img_batch, [-1, 39, 67, 67])  


Alloutput = interface(xx1)

with tf.name_scope('costFunction'):
    costFunction = -tf.reduce_sum(y * tf.log(tf.clip_by_value(Alloutput, 1e-20, 1.0)))

with tf.name_scope('train'):
    optim = tf.train.AdamOptimizer(learning_rate)
    gradient_count = optim.compute_gradients(costFunction)
    train_step = optim.apply_gradients(gradient_count)

ema = tf.train.ExponentialMovingAverage(decay=0.99)
update_losses = ema.apply([costFunction])
global_step = tf.contrib.framework.get_or_create_global_step()
incr_global_step = tf.assign(global_step, global_step + 1)
train = tf.group(update_losses, incr_global_step, train_step)
costFunction = ema.average(costFunction)
fetches["cross"] = costFunction
fetches["train"] = train

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(Alloutput, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('costFunction', costFunction)
fetches["accuracy"] = accuracy
fetches_test["accuracy"] = accuracy

merged = tf.summary.merge_all()
g_list = tf.global_variables()
saver = tf.train.Saver(var_list=g_list, max_to_keep=None)
# tf.global_variables_initializer().run()
saver.restore(sess, read_model_dir + 'model.ckpt-11')  ##

max_test = 0


coord=tf.train.Coordinator()
threads= tf.train.start_queue_runners(coord=coord)

for i in range(1000000):
    content, lab, filename ,result = sess.run([img_batch, label_batch, key_batch, fetches])
    print i, result["accuracy"], result["cross"]
    if i % 20 == 0:
        saver.save(sess, save_model_dir + 'model.ckpt', i)

coord.request_stop()
coord.join(threads)