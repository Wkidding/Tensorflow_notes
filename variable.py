# -*- coding: UTF-8 -*-
'''===============================================
@Author ：kidding
@Date   ：2020/5/24 16:37
@File   ：variable
@IDE    ：PyCharm
=================================================='''
import tensorflow as tf

#1、定义一个变量，并必须初始化
a = tf.Variable(initial_value=3.0, dtype=tf.float32)
#2、定义一个张量
b = tf.constant(value=2.0,dtype=tf.float32)
c = tf.add(a,b)

#3、变量初始化（使用全局变量初始化）
init_op = tf.global_variables_initializer()#相当于在图中间加入了一个初始化全局变量的操作
print(type(init_op))
#4、图的运行
with tf.Session() as sess:
    # 运行initial op进行变量初始化，一定要放在所有运行操作之前
    sess.run(init_op)
    #获取操作结果
    print("result :{}".format(sess.run(c)))
    print("result :{}".format(c.eval()))





