# -*- coding:utf-8 -*-
from  FaceKeyPoint import  *
import tensorflow as tf
import matplotlib.pyplot as pyplot
from matplotlib.lines import Line2D

sess = tf.InteractiveSession()
sv = tf.train.import_meta_graph("./result/model-60.meta")
sv.restore(sess,tf.train.latest_checkpoint("./result/"))

graph = tf.get_default_graph()

#print "get all names in graph:\n"
#print graph.as_graph_def()
#for node in graph.as_graph_def().node:
#    print node.name
#    print node.op
#    print "----------->"
#as_graph_def() 返回protocol buffer类型的网络定义,可以直接保存成protocol buf 类型的文件，这样就可以
#直观的看到网络节点node的输入输出以及操作，实际上就是caffe的prototext文件。已经有人做了http://blog.csdn.net/u012436149/article/details/72967540


new_x = graph.get_tensor_by_name("x_1:0") # x 已经存在了,所以顺次加_num 来标记....
new_y_ = graph.get_tensor_by_name("y_:0")
new_y_conv = graph.get_tensor_by_name("y_conv:0")
new_keep_prob = graph.get_tensor_by_name("keep_prob_1:0")


X,y = input_data(test=True)
y_pred = []

#print(X[0].reshape([96,96])*255)
#pyplot.imshow(X[0].reshape([96,96])*255)
#pyplot.show()

TEST_SIZE = X.shape[0]
for i in xrange(0,TEST_SIZE,BATCH_SIZE):
    y_batch = new_y_conv.eval(feed_dict={new_x:[X[i:i+BATCH_SIZE]],new_keep_prob:1.0})
    y_pred.extend(y_batch)

#y_batch = new_y_conv.eval(feed_dict={new_x:[X[0]],new_keep_prob:1.0}) #推理batch_size为1，送入数据的batch就是推理的batch size

##画出 print X[i]和所有y_pred[i][key_point_index[0:30]]
print "len of y_pred:", len(y_pred[0])
if len(y_pred)<TEST_SIZE*30/BATCH_SIZE:
    sys.exit(1)
show_imags=10
for i in xrange(1):
    pyplot.imshow(X[i].reshape([96, 96]) * 255,cmap=pyplot.cm.gray)
    for j in xrange(15):
        point=pyplot.plot([y_pred[i][2*j]],[y_pred[i][2*j+1]],"ro")
        #pyplot.setp(point, color='r', linewidth=10.0)
    pyplot.show()
