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


class DataGenerator(Sequence):

    def __init__(self):
        self.database = DBManagement()
        # Get caption
        self.descriptions = utils.read_caption_clean_file('Flickr8k_text/Flickr8k.cleaned.token.txt')
        self.idxtoword, self.wordtoidx, self.vocab_size = utils.map_w2id(self.descriptions.values())
        self.max_len = utils.calculate_caption_max_len(self.descriptions.values())
        # Get data from database
        self.imgs = self.database.get_image_data()

    def __len__(self):
        return (len(self.imgs)//cf.batch_size)+1

    def __getitem__(self, idx):
        X1, X2, y = list(), list(), list()
        img_dict = dict((k, v) for (k, v) in self.imgs.items()
                        if k in list(self.imgs.keys())[idx * cf.batch_size:(idx + 1) * cf.batch_size])
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


if __name__ == '__main__':
    dg = DataGenerator()
    print(dg.__getitem__(1))
