import pickle
import re
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
import gensim
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'han/data')

def remove_quotations(text):
    """
    Remove quotations and slashes from the dataset.
    """
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    return text


def remove_html(text):
    """
    Very, very raw parser to remove HTML tags from
    texts.
    """
    tags_regex = re.compile(r'<.*?>')
    return tags_regex.sub('', text)

def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in dirs:
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = remove_quotations(lines)
                lines = remove_html(lines)
                # lines = gensim.utils.simple_preprocess(lines)
                # lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)
                lines
#                 sentence = ' '.join(words)
                print(lines)
                X.append(lines)
                y.append(path)


    return X, y

if __name__ == '__main__':
    train_path = os.path.join(dir_path, 'Train_Full')
    X_data, y_data = get_data(train_path)
    pickle.dump(X_data, open(os.path.join(dir_path, "X_train.pkl"), "wb"))
    pickle.dump(y_data, open(os.path.join(dir_path, "y_train.pkl"), "wb"))
    train_path = os.path.join(dir_path, 'Test_Full')
    X_data, y_data = get_data(train_path)
    pickle.dump(X_data, open(os.path.join(dir_path, "X_test.pkl"), "wb"))
    pickle.dump(y_data, open(os.path.join(dir_path, "y_test.pkl"), "wb"))
    # X_train = pickle.load(open(os.path.join(dir_path, "X_train.pkl"),"rb"))
