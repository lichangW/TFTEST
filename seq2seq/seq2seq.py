# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .utils import get_gpu_str
import time


class HiCJ(object):
    """
	    model class 
	    Talk Robot 答句必须有一个结束符号<eos>(问句不能有),让机器告诉你它说结束了,甚至可以加上开始标记sos,padding标记等(nmt中也有)；
	    不像生成诗词歌赋，输出的长度是由人指定的,即反复将上一个输出作为输入以得到下一个输出(其实最好是以指定长度内的句号结束)；
	    采用bucket形式组织输入数据，将计算每个bucket中length操作作为graph的一部分[(5,10),(10,5),(10,20),(20,10),(20,40),(40,20),(100,100)]；
	    1. decoder 的输出控制，答句加上一个<sos>标记，告诉机器开始回答，还应该有一个max-length，因为<eos>不一定会在有限长度答句末尾出现；
	    2. decoder 以encoder的final_state作为第一步的state，decoder的第一个输入是<eos>, 在training的时候decode的其他输入是目标句子; 
	      eval和inference的时候后一个cell的输入是前一个cell的输出，循环往复直到遇到<eos>或达到max-length.
	    3. encode 的输出都被丢弃了，只有在attention-model中才使用(only use encoder_outputs in attention-based models)
	    4.  residual cell;bidirection lstem; attention model;beam-search;
	    5. 如果训练的对话前后有联系则应该将decoder上一次的final_state作为initial_state输入到下一次对话应该回improve训练结果；因为实际对话中前后是有联系的
    """

    def __init__(self, vocab_size, data_generator, **encoder_params, **decoder_params, **hyper_params):

        self.default_hidden_size = 128
        self.model = None

        self.encoder_params = encoder_params  ##  src_vob_size,src_cell_type
        self.decoder_params = decoder_params  ##  trg_vob_size
        self.hyper_params = hyper_params
        self.vocab_size = vocab_size
        self.data_generator_fucn = data_generator

        self.learn_rate = hyper_params.get("learn_rate",1e-4)
        self.clip_norm = hyper_params.get("clip_norm", 5)
        self.batch_size = hyper_params.get("batch_size", 1)
        self.embedding_size = hyper_params.get("hidden_size",
                                               self.default_hidden_size)  ##embedding_size==src_hidden_size==trg_hidden_size
        self.hidden_size = hyper_params.get("hidden_size", self.default_hidden_size)

        self.MODE_TRAINING = True
        self.MODE_EVAL = False
        self.MODE_INFERENCE = False

        tf.reset_default_graph()
        self._build_model()  ## 如果只是restore 变量，则每次都要先build model然后restore. 如果不build 模型，则需要先tf.train.import_meta_graph 载入模型

        self.saver = tf.train.Saver()

    def Training(self,max_step=100000,eval_freq=100,log_freq=10):
        ## 每次用测试集eval，若连续20次没有提高，则终止以便于早点发现问题，开始调参重新尝试

        self.sesson = tf.Session()
        step = 0
        eval_loss = 0
        eval_x,eval_y,eval_ty=self.generator_eval_data()

        with self.sesson as sess:

            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for x,y,ty in self.generator_train_data():

                step += 1
                fed = {
                    self.encoder_inputs:x,
                    self.target_outputs:ty,
                    self.decoder_inputs:y
                }

                start_t = time.time()

                #尝试将final_state连续送入encoder_state
                final_state, loss,_ =sess.run([self.final_state,self.optimizer,self.loss],
                         feed_dict=fed)

                end_t = time.time()

                print("step: {}/{} ....".format(step,max_step))
                print("loss:{:.4f} ....".format(loss))
                print("time:{}/s".format(end_t-start_t))

                if step%eval_freq==0:

                    fed = {
                        self.encoder_inputs: eval_x,
                        self.target_outputs: eval_ty,
                        self.decoder_inputs: eval_y
                    }

                    final_state, loss, _ = sess.run([self.final_state, self.optimizer, self.loss],
                                                    feed_dict=fed)
                    print("eval.....")
                    print("eval loss:{}, last loss:{}".format(loss,eval_loss))
                    if

                if step>max_step:
                    break
        self.saver()







    def Eval(self, model_dir):

        self._model_loader(model_dir)
        pass

    def Inference(self, model_dir):

        self._model_loader(model_dir)

        pass

    def _model_loader(self, model_dir):
        self.sesson = tf.Session()
        latest_ckpt = tf.train.latest_checkpoint(model_dir)
        if latest_ckpt is not None:
            self.model = self.saver.restore(self.sesson, latest_ckpt)
        else:
            raise Exception("load model failed...")

    def _build_model(self, session, model_dir):

        with tf.variable_scope("model") as scope:
            self.encoder_inputs = tf.placeholder(tf.float32, shape=tf.TensorShape([self.batch_size, None]),
                                                 name="encoder_inputs")
            self.target_outputs = tf.placeholder(tf.float32, shape=tf.TensorShape([self.batch_size, None]),
                                                 name="target_outputs")
            self.decoder_inputs = tf.placeholder(tf.float32, shape=tf.TensorShape([self.batch_size, None]),
                                                 name="decoder_inputs")

        en_output, en_state = self._build_encoder()
        self.de_output, self.final_state = self._build_decoder(en_output, en_state)

        con_seq = tf.concat(self.de_output, 1)
        bs_seq = tf.reshape(con_seq, [-1, self.hidden_size])

        with tf.variable_scope("softmax") as scope:
            soft_w = tf.Variable(
                tf.truncated_normal(shape=[self.hidden_size, self.vocab_size], dtype=tf.dtypes.float32, stddev=0.1))
            soft_b = tf.get_variable(name="soft_b", shape=[])

        self.logits = tf.matmul(bs_seq, soft_w) + soft_b
        self.predictions = tf.nn.softmax(self.logits, name="predictions")

    def _build_encoder(self):

        cell_type = self.hyper_params.get("cell_type", "BasicLSTMCell")
        layer_num = self.hyper_params.get("layer_num", 1)

        embedding_size = self.embedding_size
        vob_size = self.encoder_params.get("src_vob_size", 0)
        if vob_size == 0:
            raise Exception("unknow encode_vob_size")
        with tf.variable_scope("encoder") as scope:
            embeddings = tf.get_variable("embedding", [vob_size, embedding_size])
            encoded_input = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
            cells = self._build_cells(cell_type, self.hidden_size, layer_num, get_gpu_str(0))
            multiCell = tf.contrib.rnn.MultiRNNCell(cells)
            ## 默认一句对话与下一句对话之间无联系，因此不需要上一次对话的state作为初始状态；则很有必要将上一次的decoder最终state作为 \
            ## 下一次的encoder中
            ## self.encode_initial_state =  multiCell.zero_state(self.num_seqs, tf.float32)
            outputs, state = tf.nn.dynamic_rnn(multiCell, encoded_input, dtype=tf.dtypes.float32)
            return outputs, state

    def _build_decoder(self, encode_outputs, encode_state):

        cell_type = self.hyper_params.get("cell_type", "BasicLSTMCell")
        hidden_size = self.hyper_params.get("hidden_size", self.default_hidden_size)
        layer_num = self.hyper_params.get("layer_num", 1)

        embedding_size = self.embedding_size
        vob_size = self.encoder_params.get("trg_vob_size", 0)
        if vob_size == 0:
            raise Exception("unknow decoder_vob_size")
        with tf.variable_scope("decoder") as scope:
            embeddings = tf.get_variable("embedding", [vob_size, embedding_size])
            decoder_input = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)
            cells = self._build_cells(cell_type, hidden_size, layer_num, get_gpu_str(1))
            multiCell = tf.contrib.rnn.MultiRNNCell(cells)
            outputs, state = tf.nn.dynamic_rnn(multiCell, decoder_input, initial_state=encode_state,dtype=tf.dtypes.float32)
            return outputs, state

    def _build_loss(self):

        ## 计算loss的方法是错的？输入go,y1,y2，输出y1',y2',y3',然后计算y1,y2,y3与y1',y2',y3'之间的loss
        with tf.name_scope("loss") as scope:
            label_one_hot = tf.one_hot(self.target_outputs, self.vocab_size)
            labels = tf.reshape(label_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)
            self.loss = tf.reduce_mean(loss)

    def _build_optimizer(self):

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip_norm)
        opt=tf.train.AdamOptimizer(self.learn_rate)
        self.optimizer = opt.apply_gradients(zip(grads,tvars))

def _build_cells(self, cell_type, hidden_size, layer_num, device_id, **kwargs):
    cells = []
    Cell = None
    with tf.device(device_id):
        if cell_type == "BasicRNNCell":
            if type(hidden_size) is list and len(hidden_size) == layer_num:
                cells = [tf.contrib.rnn.BasicRNNCell(num) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.BasicRNNCell(hidden_size) for num in xrange(layer_num)]
        elif cell_type == "LSTMCell":
            if type(hidden_size) is list and len(hidden_size) == layer_num:
                cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(num, self.dropout_keep_prob) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size, self.dropout_keep_prob) for num in
                         xrange(layer_num)]

        elif cell_type == "GRUCell":
            if type(hidden_size) is list and len(hidden_size) == layer_num:
                cells = [tf.contrib.rnn.GRUCell(num) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.GRUCell(hidden_size) for num in xrange(layer_num)]
        elif cell_type == "NASCell":
            if type(hidden_size) is list and len(hidden_size) == layer_num:
                cells = [tf.contrib.rnn.NASCell(num) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.NASCell(hidden_size) for num in xrange(layer_num)]
        elif cell_type == "ConvCell":
            # convCell and gridCell for vedio process,
            if type(hidden_size) is list and len(hidden_size) == layer_num:
                cells = [tf.contrib.rnn.ConvLSTMCell(num) for num in hidden_size]
            else:
                cells = [tf.contrib.rnn.ConvLSTMCell(hidden_size) for num in xrange(layer_num)]

    return cells

