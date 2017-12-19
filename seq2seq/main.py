import  tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("convs_path","/Users/cj/workspace/TFTEST/seq2seq/chinese2english/dgk_lost_conv/results","path of conversation files which have suffix of .conv ")
flags.DEFINE_string("training_vec","data/training.vec","encoded training data")
flags.DEFINE_string("testing_vec","data/testing.vec","encoded testing data")

VOB_LIST_SIZE=5000
FLAGS=flags.FLAGS

def main():

    FLAGS.convs_path


if __name__ == '__main__':
    tf.app.run()