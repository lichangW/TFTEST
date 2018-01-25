import os,sys,codecs
import logging
import traceback
import pickle

logging.basicConfig(level=logging.DEBUG,stream=logging.StreamHandler)

PARAGRAPH_MARK='E'
CONV_MARK='M'

EOS="_EOS_" # end mark
SOS="_SOS_" # start mark
UNK="_UNK_" # unkown word
PAD="_PAD_" # padding for empty place
ALLOWED_PUNCTUATIONS=[',','?','!','.',',',';',' ']


convs_path = "/Users/cj/workspace/TFTEST/seq2seq/chinese2english/dgk_lost_conv/results"
vocab_list = "data/vocab.list"


class vocabConverter(object):

    def __init__(self,source_files, vocab_list_file, vocab_size):

        self._vocab = None
        self._vocab = None
        self._convs = None
        self._conv_vec = None
        self._word2code = None
        self._code2word = None
        self.buckets = [(5,10),(10,5),(10,20),(20,10),(20,40),(40,20),(100,100)]

        self.__init(self,source_files, vocab_list_file, vocab_size)

    def __init(self,source_files, vocab_list_file, vocab_size):

        if not os.path.exists(convs_path):
            logging.error("convs not exists!")
            return
        files = os.listdir(convs_path)  # end width .cov
        self.convs = [convs_path + "/" + conv for conv in files if conv.endswith(".conv")]

        vob_count = {}

        try:
            for conv in self.convs:
                cfile = codecs.open(conv, mode="r", encoding="utf-8")
                text = cfile.read(1e6)
                while len(text) > 0:

                    text.strip()

                    if text[0] == CONV_MARK:
                        continue


                    ### 产生conversation table [ask,pad|answer,pad]
                    text.strip(PARAGRAPH_MARK)
                    text.strip(CONV_MARK)



                    for word in text:
                        if word == PARAGRAPH_MARK or word == CONV_MARK:
                            continue
                        #if word in ALLOWED_PUNCTUATIONS or (u'\u4e00' <= word <= u'\u9fa5' :
                            # 不是在汉字范围内,  u'\u4e00'-u'\u9fff'

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

            vob_count_list.append((EOS,len(vob_count_list)))
            vob_count_list.append((SOS,len(vob_count_list)))
            vob_count_list.append((PAD,len(vob_count_list)))
            self._vocab=[x[0] for x in vob_count_list]
            self._word2code={word:code for code,word in enumerate(self._vocab)}
            self._code2word=dict(enumerate(self._vocab))

            with open(vocab_list_file,"wb") as f:
                pickle.dump(self._vocab,f)
        except Exception as _e:
            logging.error("unexpected text conv error:%s, %s ",str(_e),traceback.format_exc())
            return

    def vord2code(self,word):

        if word in self._word2code:
            return self._word2code[word]

        return len(self._vocab)

    def code2word(self,code):

        if code in self._code2word:
            return self._code2word[code]
        return self.UNK

    def text2codes(self,text):
        codes=[]
        for word in text:
            codes.append(self._word2code(word))
        return  codes

    def codes2text(self,codes):
        text=[]
        for code in codes:
            text.append(self._code2word(code))
        return text
    def vob_size(self):
        return len(self._vocab)+1

    def generate_conv_vec(self,conv_vec_path):

        try:

            for cv in self.convs:



            conv_vec = {"bucket":self.buckets,"conv":self.conv_vecs}
            with open(conv_vec_path,"wb") as f:
                pickle.dump(conv_vec,f)
        except Exception as _e:
            logging.error("generate conversation encoding vector,err:%s",str(_e),traceback.format_exc())
            return




    def batch_generator(self,batch_size):



