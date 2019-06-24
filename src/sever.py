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
from googletrans import Translator


# initialize our Flask application and the Keras model
app = flask.Flask(__name__, static_url_path="", static_folder="demo")
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['JSON_AS_ASCII'] = False

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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

@app.route("/predict", methods=["POST"])
def predict():
    # FIXME: Need front end and handle file not path
    data = {"success": False}
    error = None
    translator = Translator()

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.form.get("image") != None:

            # read the image in PIL format
            file = flask.request.form["image"]
            print('File: ', file, '\n')
            feature = vectorize_img(file)
            cap = gen_caption(feature)
            data['success'] = True
            result = translator.translate(cap, src='en', dest='vi')
            data['vi'] = result.text
            data['en'] = cap

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

@app.route('/upload', methods=['POST'])
def upload_file():
    data = dict()
    #translator = Translator()
    if flask.request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in flask.request.files:
            flash('No file part')
            return redirect(flask.request.url)
        filepath = flask.request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if filepath.filename == '':
            flash('No selected file')
            return redirect(flask.request.url)
        if filepath and allowed_file(filepath.filename):
            filename = filepath.filename
            file = 'demo/'+filename
            feature = vectorize_img(file)
            cap = gen_caption(feature)
            data['src'] = filename
            #result = translator.translate(cap, src='en', dest='vi')
            #data['vi'] = result.text
            data['Caption'] = cap
            # return flask.jsonify(data)
            return flask.render_template('result.html', result=data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_sever_model()
    app.run(debug = False, threaded = False)
    #################### Test #########################################
    #translator = Translator()
    #result = translator.translate('A tall man', src='en', dest='vi')
    #print(result.text)
    ####################################################################
