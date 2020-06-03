# -*- coding: UTF-8 -*-
'''===============================================
@Author ：kidding
@Date   ：2020/5/26 22:40
@File   ：linear_regression
@IDE    ：PyCharm
=================================================='''

import  numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#1、构造一个数据
np.random.seed(28)
N = 100
#np.linspace(0,6,N)：从0--6之间等间距的产生 N 个数据
#np.linspace(0,6,N)+np.random.normal:加上正态分布产生的随机数，目的是不让他等间距
#np.random.normal(loc = 0 , scale=2 , size=N):loc为均值，scale为标准差，size为数量
x = np.linspace(0,6,N) + np.random.normal(loc = 0 , scale=2 , size=N)
y = 14 * x - 7 + np.random.normal(loc=0.0,scale=5.0,size=N)

#将x,y设置成矩阵
x.shape = -1, 1
y.shape = -1, 1

#2、模型构建
#y = wx + b
#定义一个变量w,b
#random_uniform（shanpe,minval,maxval,name）:产生一个服从均匀分布的随机数列
#shape:产生给多少个数据/产生的数据格式(本例为1列)，minval: 均匀分布中可能出现的最小值，miaxval:均匀分布中可能出现的最大值
w = tf.Variable(initial_value=tf.random_uniform(shape = [1],minval = -1.0,maxval=1.0),name='W')
b = tf.Variable(initial_value=tf.zeros([1]),name='b')

#构建一个预测值
y_hat = w * x + b

#构建一个损失函数
#以MSE作为损失函数：预测值和实际值之间的平方和
loss = tf.reduce_mean(tf.square(y_hat-y),name='loss')

#以随机梯度下降的方式优化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05) #学习率
#在优化的过程中是让呢个函数最小化
train = optimizer.minimize(loss,name='train')

#全局变量更新
init_op = tf.global_variables_initializer()

#定义一个输出函数
def print_info(r_w,r_b,r_loss):
    print("w={},b={},loss={}".format(r_w,r_b,r_loss))

#运行
with tf.Session() as sess:
    #初始化
    sess.run(init_op)
    #输出初始化的W,b,loss
    r_w,r_b,r_loss = sess.run([w,b,loss])
    print_info(r_w,r_b,r_loss)

    #进行训练
    for step in range(100):
        #模型训练
        sess.run(train)
        # 输出训练后的W,b,loss
        r_w, r_b, r_loss = sess.run([w, b, loss])
        print_info(r_w, r_b, r_loss)