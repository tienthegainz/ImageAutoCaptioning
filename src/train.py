"""
    Train and Data preparation in here
"""
import numpy as np
from database import DBManagement
import config as cf
from word_feature import utils, word_embed
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import Sequence
from keras.layers import LSTM, Embedding,  Dense, Activation, Flatten,\
                         Reshape, concatenate, Dropout, BatchNormalization
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau,\
                            ModelCheckpoint, CSVLogger
from keras.layers.merge import add
from keras.models import Model
from keras import Input, layers
from keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers
from keras.models import load_model
from keras.initializers import glorot_uniform, he_uniform


class DataGenerator(Sequence):

    def __init__(self, mode = 'train'):
        self.database = DBManagement()
        # Get caption
        self.descriptions = utils.read_caption_clean_file('Flickr8k_text/Flickr8k.cleaned.lemma.token.txt')
        self.idxtoword, self.wordtoidx, self.vocab_size = utils.map_w2id(self.descriptions.values())
        self.max_len = utils.calculate_caption_max_len(self.descriptions.values())

        if mode not in ('train', 'val', 'test'):
            raise ValueError()
        self.mode = mode # choose data generator mode
        if self.mode == 'train':
            """Call train image vector"""
            self.imgs = self.database.get_image_data_from_list('Flickr8k_text/Flickr_8k.trainImages.txt')
        elif self.mode == 'val':
            """Call val image vector"""
            self.imgs = self.database.get_image_data_from_list('Flickr8k_text/Flickr_8k.devImages.txt')
        if self.mode == 'test':
            """Call test image vector"""
            self.imgs = self.database.get_image_data_from_list('Flickr8k_text/Flickr_8k.testImages.txt')
        #### Test purpose ####
        # self.imgs = self.database.get_image_data()

    def __len__(self):
        return len(self.imgs)//cf.batch_size

    def __getitem__(self, idx):
        X1, X2, y = list(), list(), list()

        img_dict = dict((k, v) for (k, v) in self.imgs.items()
                        if k in list(self.imgs.keys())[idx * cf.batch_size:(idx + 1) * cf.batch_size])

        # print('\n{}.Batch size: {}\n'.format(idx,len(img_dict)))
        for k, v in img_dict.items():
            desc_list = self.descriptions[k.split('.')[0]]
            ### Debug ###
            # print("Length of feature vector: {} of {}\n".format(len(v), k))
            ##############
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


def build_concat(max_length, vocab_size, str_list):
    # Image input
    inputs1 = Input(shape=(cf.vector_len, ))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(512, activation='relu')(fe1)
    # Text
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, cf.embedding_dim, mask_zero=True)(inputs2)

    se2 = Dropout(0.5)(se1)
    # Remember to change
    se3 = LSTM(512)(se2)
    # Concatenate
    decoder1 = add([fe2, se3])
    decoder2 = Dense(512, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax',
                    kernel_regularizer=regularizers.l2(0.02))(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # Only after concate, tensor become layer
    model.layers[2].set_weights([word_embed.make_word_matrix(str_list)])
    model.layers[2].trainable = False

    return model


if __name__ == '__main__':
    train_gen = DataGenerator()
    val_gen = DataGenerator(mode = 'val')

    # build or load model
    model = build_concat(train_gen.max_len, train_gen.vocab_size, train_gen.descriptions.values())
    # model = load_model('history/train_lemma.64-3.38.hdf5')
    print(model.summary())
    # compile
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)
    # opt = SGD(lr=0.1, momentum=0.8, decay=0.0005, nesterov=True)
    # opt = RMSprop(lr=0.002, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='history/train_lemma_test.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('history/train_test.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=0.001)
    # Train
    """Fit models by generator"""
    history = model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        callbacks=[csv_logger, checkpointer, reduce_lr],
                        epochs=100)
    """Plot training history"""
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('history/accuracy_lemma_test.png')

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('history/loss_lemma_test.png')
