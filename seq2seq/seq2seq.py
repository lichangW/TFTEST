import tensorflow as tf
import numpy as np

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class HiCJ(object):
    """
	    model class 
	    Talk Robot问句和答句必须有一个结束符号<eos>,让机器告诉你它说结束了,甚至可以加上开始标记sos,padding标记等(nmt中也有)；
	    不像生成诗词歌赋，输出的长度是由人指定的,即反复将上一个输出作为输入以得到下一个输出(其实最好是以指定长度内的句号结束)；
	    采用bucket形式组织输入数据，将计算每个bucket中length操作作为graph的一部分；
    """

    def __init__(self,cell_type,hidden_size,layer_num,vocab_size,data_generator):
        self.hidden_size=hidden_size
        self.layer_num=layer_num
        self.vocab_size=vocab_size
        self.data_generator_fucn=data_generator

    def _build_encoder(self):

        tf.nn.dynamic_rnn()

    def _build_decoder(self):


    def _build_cells(self,cell_type,hidden_num):

        if cell_type=="BasicRNNCell":
            cells = tf.contrib.rnn.BasicRNNCell(hidden_num)




def data_generator():
    pass


