import tensorflow as tf
import numpy as np
import os
#
list_dir = './train_dis_np_67/'
tfrecords_dir = './train_67_tfrecord/'

def getNameList():
    list=[]
    for i in range(1, 150):
        for j in range(1, 50):
            name = 'a' + str(i) + '_r' + str(j) +'.npy'
            if os.path.exists(list_dir+ name):
                list.append(name)
    return list
pass

name_list=getNameList()
length=len(name_list)
if not os.path.exists(tfrecords_dir):
    os.mkdir(tfrecords_dir)

for i in range(length):
    if i % 100 ==0:
        print i
    save_name = tfrecords_dir + name_list[i] + '.tfrecord'
    str=''
    for j in range(1,10):
        if name_list[i][j]=='_':
            break
        str=str+name_list[i][j]
    aaa= int(str)
    print aaa
    writer = tf.python_io.TFRecordWriter(save_name)
    img_raw = np.load(list_dir + name_list[i])
    img_raw = img_raw.tostring()
    label = np.zeros([140])
    label[aaa-1] = 1
    label = label.tostring()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    writer.write(example.SerializeToString())
    writer.close()

