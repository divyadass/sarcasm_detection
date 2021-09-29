from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import re

import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

import flask
app = Flask(__name__)

max_features = 10000  ## no of unique words in the vocabulary
maxlen = 15 ## no of words to use from each headline
embedding_size = 100 ## length of word embedding

######################### loading the tokenizer object to make a sequence of number from string text
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

######################### loading the trained keras model ############################

json_file = open('sarcasm_detection.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
clf = model_from_json(loaded_model_json)
# load weights into new model
clf.load_weights("sarcasm_detection.h5")

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()

    tokenized = tokenizer.texts_to_sequences([to_predict_list['review_text']])
    tokenized_pad = pad_sequences(tokenized, maxlen = maxlen, value=0.0)
    
    prob = clf.predict(tokenized_pad)
    
    if prob[0][0]>=0.5:
        prediction = "Positive"
    else:
        prediction = "Negative"        
    
    return flask.render_template('predict.html', prediction = prediction, prob =np.round(prob[0][0],3)*100)

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host='localhost', port=8081)
