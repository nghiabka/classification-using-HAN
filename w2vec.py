import pickle

from gensim.models import KeyedVectors
import os


dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
word2vec_model_path = os.path.join(dir_path, "han/model/model.vec")
print(word2vec_model_path)

w2v = KeyedVectors.load_word2vec_format(word2vec_model_path)
vocab = w2v.wv.vocab
wv = w2v.wv

def get_word2vec_data(X):
    word2vec_data = []
    for x in X:
        sentence = []
        for word in x.split(" "):
            if word in vocab:
                sentence.append(wv[word])

        word2vec_data.append(sentence)

    return word2vec_data

X_train = pickle.load(open(os.path.join(dir_path, "han/data/X_train.pkl"), "rb"))
X_test = pickle.load(open(os.path.join(dir_path, "han/data/X_test.pkl"), "rb"))

X_data_w2v = get_word2vec_data(X_train)
X_test_w2v = get_word2vec_data(X_test)

pickle.dump(X_data_w2v, open(os.path.join(dir_path, "X_train_w2v.pkl"), "wb"))
pickle.dump(X_test, open(os.path.join(dir_path, "X_test_w2v.pkl"), "wb"))
