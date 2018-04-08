

import datetime
import tensorflow as tf
import numpy as np
import mnist_net


batch_size = 32
record_file = "mnist_train.tfrecord"

#filename_queue = tf.train.string_input_producer([record_file], num_epochs=10)
filename_queue = tf.train.string_input_producer([record_file],num_epochs=100,shuffle=True)

# read and decode
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = {"label": tf.FixedLenFeature([],tf.int64),
            "image": tf.FixedLenFeature([],tf.string) }
features = tf.parse_single_example(serialized_example, features = features)
image = tf.decode_raw(features['image'],tf.uint8)
image = tf.cast(image,tf.float32)
image = tf.reshape(image,[28*28])
label = features['label']
#label = tf.reshape(label,[1])
#image = tf.reshape(image,img_shape)

#inputs = tf.placeholder(tf.float32,shape=[batch_size,784])
#labels = tf.placeholder(tf.int32,shape=[batch_size])


min_after_dequeue = 100
capacity = min_after_dequeue+3*batch_size
img_batch,lab_batch = tf.train.shuffle_batch([image,label],
                                            batch_size=batch_size,
                                            capacity = capacity, 
                                            min_after_dequeue = min_after_dequeue )

hidden1_units = 20
hidden2_units = 50

logits = mnist_net.inference(img_batch,hidden1_units,hidden2_units)

loss = mnist_net.loss(logits,lab_batch)
acc = mnist_net.evaluation(logits,lab_batch)

train_op = mnist_net.training(loss, 1e-5)

init_group = tf.group([tf.global_variables_initializer(),tf.local_variables_initializer()])


with tf.Session() as sess:
    sess.run(init_group)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)


    for i in range(200000):
        [los,ac,_] = sess.run([loss,acc,train_op]) #,feed_dict={inputs:img,labels:lab}
        if i%1000==0:
            print("iter: {} - time:{} - loss: {:<.5} - acc:{:.3%}".format(i, datetime.datetime.now(),los,ac))
