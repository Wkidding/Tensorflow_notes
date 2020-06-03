# -*- coding: UTF-8 -*-
'''===============================================
@Author ：kidding
@Date   ：2020/5/24 17:27
@File   ：feed_fetch
@IDE    ：PyCharm
=================================================='''
import tensorflow as tf
#构建矩阵乘法，但是矩阵在运行的时候给定
m1 = tf.placeholder(dtype=tf.float32,shape=[2,3],name="placeholder_m1")
m2 = tf.placeholder(dtype=tf.float32,shape=[3,2],name="placeholder_m2")
m3 = tf.matmul(m1,m2)


with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
    #通过Session的config参数可以对TensorFlow的应用的执行进行一些优化调整
    #log_device_placement ： 是否打印日志，默认为False,不打印日志
    #allow_soft_placement ： 是否允许动态使用CPU/GPU,默认为False...
    #                        当我们的安装方式为GPU时，建议修改为True(因为TensorFlow中的op部分只能在CPU上运行)
    print("result:\n{}".format(sess.run(fetches=[m3],feed_dict={m1:[[1,2,3],[4,5,6]],m2:[[9,8],[7,6],[5,4]]})))
