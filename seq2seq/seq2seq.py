import tensorflow as tf
import numpy as np

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class HiCJ(object):
    """
	    model class 
	    Talk Robot答句必须有一个结束符号<eos>(问句不能有),让机器告诉你它说结束了,甚至可以加上开始标记sos,padding标记等(nmt中也有)；
	    不像生成诗词歌赋，输出的长度是由人指定的,即反复将上一个输出作为输入以得到下一个输出(其实最好是以指定长度内的句号结束)；
	    采用bucket形式组织输入数据，将计算每个bucket中length操作作为graph的一部分[(5,10),(10,5),(10,20),(20,10),(20,40),(40,20),(100,100)]；
	    1.decoder的输出控制，答句加上一个<sos>标记，告诉机器开始回答，还应该有一个max-length，因为<eos>不一定会在有限长度答句末尾出现；
	    2.decoder以encoder的final_state作为第一步的state，decoder的第一个输入是<eos>, 在training的时候decode的其他输入是目标句子; 
	      eval和inference的时候后一个cell的输入是前一个cell的输出，循环往复直到遇到<eos>或达到max-length.
	    3.encode的输出都被丢弃了，只有在attention-model中才使用(only use encoder_outputs in attention-based models)
    """

    def __init__(self,cell_type,hidden_size,layer_num,vocab_size,data_generator,**high):
        self.hidden_size=hidden_size
        self.layer_num=layer_num
        self.vocab_size=vocab_size
        self.data_generator_fucn=data_generator

    def Training(self):
        pass
    def Eval(self):
        pass
    def Inference(self):
        pass
    def Build_model(self):
        pass
    def _build_encoder(self):

        cell
        tf.nn.dynamic_rnn()

    def _build_decoder(self):
        pass

    def _build_cells(self,cell_type,hidden_num,layer_num=1,**kwargs):

        cells=[]
        Cell=None
        if cell_type == "BasicRNNCell":
            if type(hidden_num) is list and len(hidden_num)==layer_num:
                cells = [tf.contrib.rnn.BasicRNNCell(num) for num in hidden_num]
            else:
                cells = [tf.contrib.rnn.BasicRNNCell(hidden_num) for num in xrange(layer_num) ]
        elif  cell_type == "BasicLSTMCell":
            if type(hidden_num) is list and len(hidden_num)==layer_num:
                cells = [tf.contrib.rnn.BasicLSTMCell(num) for num in hidden_num]
            else:
                cells = [tf.contrib.rnn.BasicLSTMCell(hidden_num) for num in xrange(layer_num) ]

        elif cell_type == "GRUCell":
            if type(hidden_num) is list and len(hidden_num)==layer_num:
                cells = [tf.contrib.rnn.GRUCell(num) for num in hidden_num]
            else:
                cells = [tf.contrib.rnn.GRUCell(hidden_num) for num in xrange(layer_num) ]
        elif cell_type == "NASCell":
            if type(hidden_num) is list and len(hidden_num)==layer_num:
                cells = [tf.contrib.rnn.NASCell(num) for num in hidden_num]
            else:
                cells = [tf.contrib.rnn.NASCell(hidden_num) for num in xrange(layer_num) ]
        elif cell_type == "ConvCell":
            #for vedio process
            if type(hidden_num) is list and len(hidden_num)==layer_num:
                cells = [tf.contrib.rnn.ConvLSTMCell(num) for num in hidden_num]
            else:
                cells = [tf.contrib.rnn.ConvLSTMCell(hidden_num) for num in xrange(layer_num) ]

        return  cells

def data_generator():
    pass


