# -*- coding: UTF-8 -*-
'''===============================================
@Author ：kidding
@Date   ：2020/5/25 17:26
@File   ：Demo3
@IDE    ：PyCharm
=================================================='''
import tensorflow as tf
#TensorFlow变量作用域

#===============================================
# #方式一
# def my_func(x):
#     w1 = tf.Variable(tf.random_normal([1]))[0]
#     b1 = tf.Variable(tf.random_normal([1]))[0]
#     r1 = w1 * x + b1
#
#     w2 = tf.Variable(tf.random_normal([1]))[0]
#     b2 = tf.Variable(tf.random_normal([1]))[0]
#     r2 = w2 * x + b2
#
#     return r1,w1,b1,r2,w2,b2
#
# #执行
# #下面两行代码还是属于图的构建
# x = tf.constant(3,dtype=tf.float32)
# r = my_func(x)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
#     # 初始化
#     tf.global_variables_initializer().run()
#     # 执行结果
#     print(sess.run(r))

#===============================================
# #方式二
# def my_func(x):
#     #name:名称，shape:形状，initializer：初始化器（用什么方式初始化）
#
#     #tf.get_variable常用的initializer初始化器
#     # tf.random_normal_initializer(mean,stddev):初始化均值为mean，方差为stddev的服从高斯分布的随机值
#     # tf.constant_initializer(value):初始化为给定的常数值value
#     # tf.random_uniform_initializer(a,b):初始化为从a到b的均匀分布的随机值
#     # tf.orthogonal_initializer(gini=1.0):初始化是一个正交矩阵，gini作用：最终返回的矩阵是随机矩阵乘以gini的结果
#     # tf.identity_initializer(gini=1.0):初始化是一个单位矩阵，gini作用：最终返回的矩阵是随机矩阵乘以gini的结果
#
#     w = tf.get_variable(name='w',shape=[1],initializer=tf.random_normal_initializer())[0]
#     b = tf.get_variable(name='b',shape=[1],initializer=tf.random_normal_initializer())[0]
#     r = w * x + b
#
#     return r,w,b
#
# def func(x):
#     with tf.variable_scope('op1',reuse=tf.AUTO_REUSE):
#         r1 = my_func(x)
#     with tf.variable_scope('op2', reuse=tf.AUTO_REUSE):
#         r2 = my_func(r1[0])
#     return r1,r2
# #执行
# #下面几行代码还是属于图的构建
# x1 = tf.constant(3,dtype=tf.float32)
# x2 = tf.constant(5,dtype=tf.float32)
# r1 = func(x1)
# r2 = func(x2)
#
# with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
#     # 初始化
#     tf.global_variables_initializer().run()
#     # 执行结果
#     print(sess.run(r1))
#     print(sess.run(r2))

#===============================================
# #可视化
# with tf.device("/gpu:0"):
#     with tf.variable_scope("foo"):
#         x_initl = tf.get_variable("init_x" , [10] , tf.float32 , initializer = tf.random_normal_initializer())[0]
#         x = tf.Variable(initial_value=x_initl,name='x')
#         y = tf.placeholder(dtype=tf.float32,name='y')
#         z = x + y
#
#     with tf.variable_scope("bar"):
#         a = tf.constant(3.0)+4.0
#
#     w =  z*a
# #开始记录信息(需要展示的信息的输出)
# tf.summary.scalar('scalar_init_x',x_initl)
# tf.summary.scalar(name='scalar_init_x',tensor=x)
# tf.summary.scalar('scalar_y',y)
# tf.summary.scalar('scalar_z',z)
# tf.summary.scalar('scalar_w',w)
#
# # 更新操作:x
# assign_op = tf.assign(x, x + 1)
# with tf.control_dependencies([assign_op]):
#     with tf.device('/gpu:0'):
#         out = x * y
#     tf.summary.scalar('scala_out',out)
#
# #运行
# with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
#     merged_summary = tf.summary.merge_all()
#     writer = tf.summary.FileWriter('./result',sess.graph)
#     #初始化
#     sess.run(tf.global_variables_initializer())
#     # 执行结果
#     for i in range (1,5):
#         summary,r_out,r_x,r_w = sess.run([merged_summary,out,x,w],feed_dict={y:1})
#         writer.add_summary(summary,i)
#         print("{},{},{}".format(r_out,r_x,r_w))
#
#     #关闭操作
#     writer.close()
#
# #打开网页可视化步骤：
# # 1、运行.py文件
# # 2、在result中点击open in Terminal
# # 3、输入指令 tensorboard --logdir=文件位置（打开pycharm工作区，找到result文件，复制位置即可）
# # 4、点击网址，将 http://LAPTOP-5NN1OKPJ:6006/中间的 LAPTOP-5NN1OKPJ 改为 localhost 即可

#===============================================
# #模型保存
# v1 = tf.Variable(tf.constant(7.0),name = 'v1')
# v2 = tf.Variable(tf.constant(5.0),name = 'v2')
# result = v1+v2
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     #模型保存到model文件夹下，文件前缀为：model.one
#     saver.save(sess,'./model/model.one')
#     #model.one.meta:保存模型的数据信息
#     #model.one.index:保存模型的索引
#     #chackpoint:检查文件
#     #model.one.data-00000-of-00001:保存模型中的数据

#===============================================
# #模型的提取：1、完整提取：需要完整恢复保存之前的数据格式
# #此处的变量名必须和保存时候的变量名一致
# v1 = tf.Variable(tf.constant(2.0),name = 'v1')
# v2 = tf.Variable(tf.constant(4.0),name = 'v2')
# result = v1+v2
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     #会从对应的文件中加载变量，图等相关信息，所以上面1.0和2.0是没用的，他们的值是在保存模型时候的值，也就是说输出是7.0+5.0
#     saver.restore(sess,'./model/model.one')
#     print(sess.run(result))

#===============================================
# #2、直接加载图，不需要定义变量
# saver = tf.train.import_meta_graph('./model/model.one.meta')
#
# with tf.Session() as sess:
#     saver.restore(sess,'./model/model.one')
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
# ===============================================

# #3、加载模型的时候给定变量的映射关系
# a = tf.Variable(tf.constant(2.0),name = 'a')
# b = tf.Variable(tf.constant(4.0),name = 'b')
# result = a+b
#
# saver = tf.train.Saver({"v1":a,"v2":b})
# #意为加载模型的时候把模型本来有的v1,v2用a,b进行重命名，也就是说v1->a,v2->b是一组映射关系
#
# with tf.Session() as sess:
#     saver.restore(sess,'./model/model.one')
#     print(sess.run(result))
# ===============================================
