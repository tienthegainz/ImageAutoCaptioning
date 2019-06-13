import sys
import config as cf
from src.word_feature.utils import get_unique_word, read_caption_clean_file, map_w2id
sys.path.append('../')


def load_glove(path):
    """
        Give you the dict of word and its coefficent
    """
    f = open(path, encoding='utf-8')
    print("Loading the /{}/ vector".format(path.split('/')[-1]))
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def make_word_matrix(str_list):
    # FIXME
    """
        Give you word customized matrix
    """
    idxtoword, wordtoidx, vocab_size = map_w2id(str_list)
    embedding_matrix = np.zeros((vocab_size, cf.embedding_dim))
    for word, i in wordtoidx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    # Find what to return #
    return

