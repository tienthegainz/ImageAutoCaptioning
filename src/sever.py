from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import flask
import io
import config as cf

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
caption_model = None
vector_model = None

def load_sever_model():
    # Load caption model and vector
    global caption_model
    global vector_model

    caption_model = load_model('history/train_lemma.64-3.38.hdf5')

    model = InceptionV3(weights='imagenet')
    vector_model = Model(model.input, model.layers[-2].output)

def vectorize_img(file):
    img = image.load_img(file, target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = vector_model.predict(x)
    # flatten
    features = np.reshape(features, cf.vector_len)
    return features


@app.route("/predict", methods=["POST"])
def predict():
    # FIXME
    data = {"success": False}
    error = None

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
            img = image.load_img(file, target_size)


    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_sever_model()
    app.run(debug = False, threaded = False)
