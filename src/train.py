"""
    Train and Data preparation in here
"""
import numpy as np
from database import DBManagement
import config as cf
from word_feature import utils, word_embed
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.models import Model
from keras import Input, layers
from keras import optimizers



class DataGenerator(Sequence):

    def __init__(self):
        self.database = DBManagement()
        # Get caption
        self.descriptions = utils.read_caption_clean_file('Flickr8k_text/Flickr8k.cleaned.token.txt')
        self.idxtoword, self.wordtoidx, self.vocab_size = utils.map_w2id(self.descriptions.values())
        self.max_len = utils.caculate_caption_max_len(self.descriptions.values())
        # Get data from database
        """
            I want a dict of key: img_name and value: corresponding vector
        """
        self.imgs = self.database.get_database

    def __len__(self):
        return (len(self.imgs)//cf.batch_size)+1

    def __getitem__(self, idx):
        X1, X2, y = list(), list(), list()
        img_dict = dict([k, v for k, v in self.imgs.items()
                    if k in self.imgs.keys[idx * self.batch_size:(idx + 1) * self.batch_size]])
        for k, v in img_dict.items():
            desc_list = self.descriptions[k.split('.')[0]]
            for desc in desc_list:
                seq = [self.wordtoidx[word] for word in desc.split(' ')]
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=self.max_len)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    # store
                    X1.append(v)
                    X2.append(in_seq)
                    y.append(out_seq)
        return [[np.array(X1), np.array(X2)], np.array(y)]
