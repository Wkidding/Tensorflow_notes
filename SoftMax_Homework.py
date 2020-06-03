# -*- coding: UTF-8 -*-
'''===============================================
@Author ：kidding
@Date   ：2020/5/27 11:55
@File   ：SoftMax_Homework
@IDE    ：PyCharm
=================================================='''

import tensorflow as tf

#使用内建的函数加载MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MINIST_data',one_hot=True)

#创建一个交互式会话
sess = tf.InteractiveSession()

#创建占位符(placeholder)
x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

#模型构建,分配权重和偏置得到空的张量
w = tf.Variable(initial_value=tf.zeros([784,10],dtype=tf.float32))
b = tf.Variable(initial_value=tf.zeros([10],dtype=tf.float32))

#全局变量更新
sess.run(tf.global_variables_initializer())

#Softmax回归：调用Softmax函数得到概率值
y = tf.nn.softmax(tf.matmul(x,w) + b)
#构建一个损失函数
loss = tf.reduce_mean(-tf.reduce_mean(y_*tf.log(y),reduction_indices=[1],name='loss'))
#以随机梯度下降的方式优化损失函数
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss) #学习率

#训练批大小的设置
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})

#测试
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})*100
print("Softmax预测模型最终准确率是: {} % ".format(acc))

#关闭会话
sess.close()