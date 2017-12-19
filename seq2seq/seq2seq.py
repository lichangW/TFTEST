import tensorflow as tf
import numpy as np

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class HiCJ(object):
    """
	    model class 
    """

    def __init__(self,cell_type,hidden_size,layer_num,vocab_size,data_generator):
        self.hidden_size=hidden_size
        self.layer_num=layer_num
        self.vocab_size=vocab_size
        self.data_generator_fucn=data_generator

    def _build_encoder(self):

        tf.nn.dynamic_rnn()

    def _build_decoder(self):


    def _build_cells(self,cell_type,cell_num):

        if cell_type=="BasicRNNCell":

            for
            cells = ftf.contrib.rnn.BasicRNNCell()




def data_generator():
    pass


