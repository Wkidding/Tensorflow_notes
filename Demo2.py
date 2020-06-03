# -*- coding: UTF-8 -*-
'''===============================================
@Author ：kidding
@Date   ：2020/5/25 10:36
@File   ：Demo2
@IDE    ：PyCharm
=================================================='''
import tensorflow as tf

# #需求1：实现一个累加器，并且每一步输入累加器的结果值
# #1、定义一个变量
# x = tf.Variable(1,dtype=tf.float32,name='v_x')
#
# #2、变量的更新
# #x = x+1 #这一步相当于是赋值，而不是真正的更新操作
# assign_op = tf.assign(ref = x,value = x + 1 )#ref:指定要更新的变量 value：要更新的值是的多少
#
# #3、变量初始化
# x_init_op = tf.global_variables_initializer()
#
# #4、运行
# with tf.Session(config= tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
#     #变量初始化
#     sess.run(x_init_op)
#     #模拟迭代更新累加器
#     for i in range(5):
#         r_x = sess.run(x)
#         print(r_x)
#         #执行更新操作
#         sess.run(assign_op)

# #需求二：实现动态更新变量的维度数目
# #1、定义一个不定形状的变量
# x = tf.Variable(initial_value=[], #给定一个空值
#                 dtype=tf.float32,
#                 trainable = False, #要是定义一个不固定形状的值，将该参数设为False
#                 validate_shape=False  #设置为True表示变量在更新的时候进行Shape检查，默认为True
#                 )
#
# #2、变量更改
# concat = tf.concat([x,[0.0,0.0]],axis=0)    #axis=0为行向量，axis=1为列向量
# assign_op = tf.assign(x,concat,validate_shape=False)
#
# #3、变量初始化
# x_init_op = tf.global_variables_initializer()
#
# #4、运行
# with tf.Session(config= tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
#     #变量初始化
#     sess.run(x_init_op)
#     #模拟迭代更新累加器
#     for i in range(5):
#         r_x = sess.run(x)
#         print(r_x)
#         #执行更新操作
#         sess.run(assign_op)

#需求三：实现阶乘
#1、定义一个变量
sum = tf.Variable(1,dtype=tf.int32)
#2、定义一个占位符
i = tf.placeholder(dtype=tf.int32)
#3、更新操作
tmp_sum = sum * i
assign_op = tf.assign(sum,tmp_sum)#assign是更新的API

with tf.control_dependencies([assign_op]):
    #如果要执行这个代码块中的内容，必须先执行control_dependenecies中给定的操作/Tensor
    sum = tf.Print(sum,data=[sum,sum.read_value()],message="sum:")

#4、变量初始化及运行
x_init_op = tf.global_variables_initializer()
#5、运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
    #变量初始化
    sess.run(x_init_op)
    #模拟迭代更新累加器
    for j in range(1,6): #范围是1~6之间
        #执行更新操作
        #sess.run(assign_op,feed_dict={i:j})
        #通过control_dependenecies可以指定依赖关系，这样的话，就不用管内部的更新操作了
        r = sess.run(sum,feed_dict={i:j})

    print("5!={}".format(r))