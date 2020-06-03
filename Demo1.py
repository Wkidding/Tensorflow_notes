import tensorflow as tf

#1、定义常量a、b
a = tf.constant([[1,2],[3,4]],dtype=tf.int32 , name = 'a')#dtype指的是数据类型
print(type(a))
print("变量a是否在默认图中:{}".format(a.graph is tf.get_default_graph()))

b = tf.constant([5,6,7,8],dtype=tf.int32,shape=[2,2])#shape指的是格式，此处为两行两列

#2、以a、b为输入，进行乘法操作
c = tf.matmul(a,b, name= 'matmul')
print(type(c))
print(c)

'''
#使用新定义图
graph = tf.Graph()
with graph.as_default():
    #在改代码块中，图是自己新创建的，出了这个代码块，系统会使用默认图
    d = tf.constant(5.0)
    print("变量d是否在新创建图graph中:{}".format(d.graph is graph))
print("变量d是否在默认图中:{}".format(d.graph is tf.get_default_graph()))

with tf.Graph().as_default() as g2:
    e = tf.constant(6.0)
    print("变量e是否在新创建图g2中:{}".format(e.graph is g2))

'''

#3、以a、c为基础，进行相加操作
g = tf.add(a,c, name='add')
print(type(g))
print(g)

'''
以下代码错误，不能使用两个图中的变量进行操作
f = tf.add(d,e)
print("变量f的图：{}".format(f.graph))

报错信息如下：
ValueError: Tensor("Const:0", shape=(), dtype=float32) must be from the same graph as Tensor("Const:0", shape=(), dtype=float32).

'''
#4、添加剑法操作
h = tf.subtract(b,a, name='b-a')
l = tf.matmul(h,c)
r = tf.add(g,l)

#4、创建&启动会话
sess = tf.Session()
print(sess)
#调用sess的Run方法，执行矩阵乘法，得到c的结果（所以将c作为参数传入）
result= sess.run(r)
print("type:{},value:{}".format(type(result),result))
#计算时不需要考虑图中间的运算，只需要关注最终结果的对应的对象以及所需要的输入数据值
#只需要传递进去所需要得到的结果对象，会自动根据途中的依赖关系触发所有相关的op操作的执行
#fetches:表示从图中获取某个op操作的值
result2= sess.run(fetches= [r,c])
print("type:{},value:{}".format(type(result2),result2))


#关闭会话，关闭之后不能再执行sess操作
sess.close()

#第二种打开会话的方式，使用with语句块会在使用完毕后自动关闭session
with tf.Session() as sess2:

#sess2 = tf.Session()
#with sess2.as_default():

    print(sess2)

    #通过session的run方法获取张量c的结果
    print("sess2 run:{}".format(sess2.run(c)))
    # 通过张量对象的eval（）方法获取张量c的结果
    print("c eval:{}".format(c.eval()))