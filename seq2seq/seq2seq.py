import tensorflow as tf
import numpy as np

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class HiCJ(object):
    """
	    model class 
	    Talk Robot 答句必须有一个结束符号<eos>(问句不能有),让机器告诉你它说结束了,甚至可以加上开始标记sos,padding标记等(nmt中也有)；
	    不像生成诗词歌赋，输出的长度是由人指定的,即反复将上一个输出作为输入以得到下一个输出(其实最好是以指定长度内的句号结束)；
	    采用bucket形式组织输入数据，将计算每个bucket中length操作作为graph的一部分[(5,10),(10,5),(10,20),(20,10),(20,40),(40,20),(100,100)]；
	    1.decoder 的输出控制，答句加上一个<sos>标记，告诉机器开始回答，还应该有一个max-length，因为<eos>不一定会在有限长度答句末尾出现；
	    2.decoder 以encoder的final_state作为第一步的state，decoder的第一个输入是<eos>, 在training的时候decode的其他输入是目标句子; 
	      eval和inference的时候后一个cell的输入是前一个cell的输出，循环往复直到遇到<eos>或达到max-length.
	    3.encode 的输出都被丢弃了，只有在attention-model中才使用(only use encoder_outputs in attention-based models)
    """

    def __init__(self,vocab_size,data_generator,**encoder_params,**decoder_params,**hyper_params):

        self.default_hidden_size = 128
        self.model=None

        self.encoder_params = encoder_params  ##  src_vob_size,src_cell_type
        self.decoder_params = decoder_params  ##  trg_vob_size
        self.hyper_params = hyper_params
        self.vocab_size=vocab_size
        self.data_generator_fucn=data_generator

        self.batch_size = hyper_params.get("batch_size",1)
        self.embedding_size = self.hyper_params.get("hidden_size",self.default_hidden_size) ##embedding_size==src_hidden_size==trg_hidden_size

        self.MODE_TRAINING=True
        self.MODE_EVAL=False
        self.MODE_INFERENCE=False

        self.Build_model()

        self.saver = tf.train.Saver(tf.global_variables())

    def Training(self):
        self.Build_model()
        pass
    def Eval(self):
        pass
    def Inference(self):
        pass
    def Build_model(self,session,model_dir):

        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        if latest_ckpt is not None:
            self.model=self.saver.restore(session,latest_ckpt)

        with tf.variable_scope("model") as scope:
            if self.model is not None:
                self.encoder_inputs = tf.get_variable("encoder_inputs", shape=tf.TensorShape([self.batch_size, None]),
                                                  dtype=tf.float32)
            if (self.MODE_EVAL or self.MODE_INFERENCE) and self.model is None:
                raise Exception("no available model restored")
            if self.MODE_TRAINING:
                    self.encoder_inputs = tf.placeholder(tf.float32,shape=tf.TensorShape([self.batch_size,None]),name="encoder_inputs")
                    self.encoder_sequence_length = tf.placeholder(tf.float32,shape=tf.TensorShape([self.batch_size]),name="encoder_sequence_length")
                    self.decoder_inputs = tf.placeholder(tf.float32,shape=tf.TensorShape([self.batch_size,None]))
                    self.decoder_sequence_length =  tf.placeholder(tf.float32,shape=tf.TensorShape([self.batch_size]))

                    session.run(tf.local_variables_initializer())
                    session.run(tf.global_variables_initializer())
                    session.run(tf.tables_initializer())

    def _build_encoder(self):
        embedding_size = self.embedding_size
        vob_size = self.encoder_params.get("src_vob_size",0)
        if vob_size ==0:
            raise Exception("unknow encode_vob_size")
        with tf.variable_scope("encoder") as scope:
            embeddings=tf.get_variable("embedding",[vob_size,embedding_size])
            encoded_input=tf.nn.embedding_lookup(embeddings,self.encoder_inputs)
            cells = self._build_cells(self.hyper_params)
            multiCell = tf.contrib.rnn.MultiRNNCell(cells)
            outputs, state = tf.nn.dynamic_rnn(multiCell,encoded_input,sequence_length=self.encoder_sequence_length,dtype=tf.dtypes.float32)
            return outputs,state

    def _build_decoder(self,encode_outputs,encode_state):

        embedding_size = self.embedding_size
        vob_size = self.encoder_params.get("trg_vob_size",0)
        if vob_size==0:
            raise Exception("unknow decoder_vob_size")
        with tf.variable_scope("decoder") as scope:
            embeddings=tf.get_variable("embedding",[vob_size,embedding_size])
            decoder_input=tf.nn.embedding_lookup(embeddings,self.decoder_inputs)
            cells = self._build_cells(self.hyper_paramse)
            multiCell = tf.contrib.rnn.MultiRNNCell(cells)
            outputs,state = tf.nn.dynamic_rnn(multiCell,decoder_input,initial_state=encode_state ,sequence_length=self.decoder_sequence_length,dtype=tf.dtypes.float32)
            return outputs,state

    def _build_cells(self,cell_type,hidden_size,layer_num,**kwargs):

        cell_type = self.encoder_params.get("src_cell_type","BasicLSTMCell")
        hidden_size = self.encoder_params.get("src_hidden_size",self.default_hidden_size)
        layer_num = self.encoder_params.get("src_layer_num",1)

        cells=[]
        Cell=None
        if cell_type == "BasicRNNCell":
            if type(hidden_size) is list and len(hidden_size)==layer_num:
                cells = [tf.contrib.rnn.BasicRNNCell(num) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.BasicRNNCell(hidden_size) for num in xrange(layer_num) ]
        elif  cell_type == "LSTMCell":
            if type(hidden_size) is list and len(hidden_size)==layer_num:
                cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(num,self.dropout_keep_prob) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size,self.dropout_keep_prob) for num in xrange(layer_num) ]

        elif cell_type == "GRUCell":
            if type(hidden_size) is list and len(hidden_size)==layer_num:
                cells = [tf.contrib.rnn.GRUCell(num) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.GRUCell(hidden_size) for num in xrange(layer_num) ]
        elif cell_type == "NASCell":
            if type(hidden_size) is list and len(hidden_size)==layer_num:
                cells = [tf.contrib.rnn.NASCell(num) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.NASCell(hidden_size) for num in xrange(layer_num) ]
        elif cell_type == "ConvCell":
            # convCell and gridCell for vedio process,
            if type(hidden_size) is list and len(hidden_size)==layer_num:
                cells = [tf.contrib.rnn.ConvLSTMCell(num) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.ConvLSTMCell(hidden_size) for num in xrange(layer_num) ]

        return  cells

def data_generator():
    pass


