from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import flask
import io
import config as cf
from word_feature import utils


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
caption_model = None
vector_model = None
descriptions = utils.read_caption_clean_file('Flickr8k_text/Flickr8k.cleaned.lemma.token.txt')
idxtoword, wordtoidx, vocab_size = utils.map_w2id(descriptions.values())
max_len = utils.calculate_caption_max_len(descriptions.values())

def load_sever_model():
    # Load caption model and vector
    global caption_model
    global vector_model

    caption_model = load_model('history/train_lemma.64-3.38.hdf5')

    model = InceptionV3(weights='imagenet')
    vector_model = Model(model.input, model.layers[-2].output)

def gen_caption(feature):
    start = wordtoidx['startseq']
    end = wordtoidx['endseq']
    current = [start]
    length = 0
    caption = ''
    feature = np.expand_dims(feature, axis=0)
    while current[-1]!=end and length <= max_len:
        in_seq = pad_sequences([current], maxlen=max_len)[0]
        in_seq = np.expand_dims(in_seq, axis=0)
        y = caption_model.predict([feature, in_seq])
        y = np.argmax(y)
        caption = caption + ' ' + idxtoword[y]
        current.append(y)
        length += 1

    return caption.rsplit(' ', 1)[0]

def vectorize_img(file):
    img = image.load_img(file, target_size=cf.img_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    feature = vector_model.predict(x)
    # flatten
    feature = np.reshape(feature, cf.vector_len)
    return feature


@app.route("/predict", methods=["POST"])
def predict():
    # FIXME: Need front end and handle file not path
    data = {"success": False}
    error = None

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.form.get("image") != None:

            # read the image in PIL format
            file = flask.request.form["image"]
            print('File: ', file, '\n')
            feature = vectorize_img(file)
            cap = gen_caption(feature)
            data['success'] = True
            data['caption'] = cap

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_sever_model()
    app.run(debug = False, threaded = False)
    #################### Test #########################################
    #feature = vectorize_img('Flicker8k_Dataset/667626_18933d713e.jpg')
    #caption = gen_caption(feature)
    #print(caption)
    ####################################################################
