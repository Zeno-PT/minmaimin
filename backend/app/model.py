from pythainlp import word_vector
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.models import *
import tensorflow as tf
import numpy as np
import pandas as pd

class DataCleaner:
    @staticmethod
    def preprocess(st, word2vec, max_len):
        '''
        text -> tokenize -> word2vec -> embeddings
        '''
        ws = word_tokenize(st, engine='deepcut')
        x = []
        for w in ws:
            try:
                x.append(word2vec[w])
            except:
                x.append(np.zeros((300,)))
        for i in range(max(0,max_len-len(ws))):
            x.append(np.zeros((300,)))
        x = np.vstack(x)
        return x

    @staticmethod
    def process_query(st, model, word2vec, max_len=75):
        '''
        Perform model prediction on string st
        '''
        x = np.expand_dims(DataCleaner.preprocess(st, word2vec, max_len), 0)
        out = model.predict(x)[0, 1]
        return out


class MinmaiminModel:
    def __init__(self):
        self.model = load_model('../model_files/minmaimin_LSTM.h5')
        # self.deka_file = pd.read_csv("../model_files/dataframe.csv")
        self.max_len = 100
        self.word2vec = word_vector.get_model()

    def predict(self, sentence):
        word2vec = word_vector.get_model()
        output = DataCleaner.process_query(sentence, self.model, self.word2vec, self.max_len)
        return str(round(output*100,2))+'%'

model_obj = MinmaiminModel()
