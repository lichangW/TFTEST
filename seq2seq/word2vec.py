import os,sys,codecs
import logging
import traceback
import pickle

logging.basicConfig(level=logging.DEBUG,stream=logging.StreamHandler)

convs_path = "/Users/cj/workspace/TFTEST/seq2seq/chinese2english/dgk_lost_conv/results"
vocab_list = "data/vocab.list"

class vocabConverter(object):

    def __init__(self,source_files, vocab_list_file, vocab_size):

        if not os.path.exists(convs_path):
            logging.error("convs not exists!")
            return
        files = os.listdir(convs_path)  # end width .cov
        convs = [convs_path + "/" + conv for conv in files if conv.endswith(".conv")]

        vob_count = {}

        try:
            for conv in convs:
                cfile = codecs.open(conv, mode="r", encoding="utf-8")
                text = cfile.read(1e6)
                while len(text) > 0:
                    for word in text:
                        if vob_count.get(word,0)==0:
                            vob_count[word]=1
                            continue
                        vob_count[word]+=1
                    text = cfile.read(1e6)
                cfile.close()

            logging.info("statistic vocab size: %d",len(vob_count))
            vob_count_list=[]
            for word in vob_count:
                vob_count_list.append((word,vob_count[word]))
            vob_count_list.sort(key=lambda x:x[1],reverse=True)

            if len(vob_count_list) > vocab_size:
                vob_count_list=vob_count_list[:vocab_size]

            self._vocab=[x[0] for x in vob_count_list]
            self._convs=convs
            self._word2code={word:code for code,word in enumerate(self._vocab)}
            self._code2word=dict(enumerate(self._vocab))

            with open(vocab_list_file,"wb") as f:
                pickle.dump(convs,f)
        except Exception as _e:
            logging.error("unexpected text conv error:%s ",str(_e),traceback.format_exc())
            return

    def vord2code(self,word):


        pass
    def code2word(self):
        pass

    def text2codes(self,text):
        pass

    def batch_generator(self,batch_size):
        pass


