"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import pickle
import random
import string
from flask import Flask, request, render_template
import torch
import torch.nn.functional as F
import csv
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

from keras_han.model import HAN

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_FOLDER = "flask_images"
rand_str = lambda n: "".join([random.choice(string.ascii_letters + string.digits) for _ in range(n)])

model = None
word2vec = None
max_length_sentences = 0
max_length_word = 0
num_classes = 0
categories = None


@app.route("/")
def home():
    return render_template("main.html")

@app.route("/input")
def new_input():
    return render_template("input.html")

@app.route("/show", methods=["POST"])
def show():
    global han_model, embedding_matrix, word_tokenizer, MAX_WORDS_PER_SENT, MAX_SENT, MAX_VOC_SIZE,\
        GLOVE_DIM, categories
    MAX_WORDS_PER_SENT = 120
    MAX_SENT = 25
    MAX_VOC_SIZE = 300600
    GLOVE_DIM = 400
    embedding_matrix = pickle.load(open("./model/embedding_matrix.pkl", "rb"))
    word_tokenizer = pickle.load(open("./model/word_tokenizer.pkl", "rb"))
    han_model = HAN(
        MAX_WORDS_PER_SENT, MAX_SENT, 10, embedding_matrix,
        word_encoding_dim=400, sentence_encoding_dim=200
    )
    han_model.load_weights("./model/model.hdf5")
    categories = pickle.load(open("./model/classes.pkl", "rb"))
    return render_template("input.html")


@app.route("/result", methods=["POST"])
def result():
    global han_model, embedding_matrix, word_tokenizer, MAX_WORDS_PER_SENT, MAX_SENT, MAX_VOC_SIZE, \
        GLOVE_DIM, categories
    text = request.form["message"]

    x_test = [text]
    X_test_num = np.zeros((len(x_test), MAX_SENT, MAX_WORDS_PER_SENT), dtype='int32')

    for i, review in enumerate(x_test):
        sentences = sent_tokenize(review)
        tokenized_sentences = word_tokenizer.texts_to_sequences(
            sentences
        )
        tokenized_sentences = pad_sequences(
            tokenized_sentences, maxlen=MAX_WORDS_PER_SENT
        )

        pad_size = MAX_SENT - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:MAX_SENT]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )

        # Store this observation as the i-th observation in
        # the data matrix
        X_test_num[i] = tokenized_sentences[None, ...]

    result = han_model.predict(X_test_num)
    id = np.argmax(result[0])
    prob = "{:.2f} %".format(float(result[0][id])*100)
    print(prob)
    category = categories[id]

    return render_template("result.html", text=text, value=prob, index=category)


if __name__ == "__main__":
    # app.secret_key = os.urandom(12)
    app.run(host="0.0.0.0", port=4555, threaded=False)
