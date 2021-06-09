from flask import Flask, render_template, request
from pythainlp.tokenize import word_tokenize
from pythainlp import word_vector
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
def preprocess(st, word2vec, max_len):
    '''
    text -> tokenize -> word2vec -> embeddings
    '''
    ws = word_tokenize(st, engine='newmm')
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

def process_query(st, model, word2vec, max_len=75):
    '''
    Perform model prediction on string st
    '''
    x = np.expand_dims(preprocess(st, word2vec, max_len), 0)
    out = model.predict(x)[0, 1]
    return out

def get_model(num_label, max_len):
    inp = Input(shape=(max_len, 300))
    x = inp
    for i in range(3):
        x = Bidirectional(LSTM(50, return_sequences=True, recurrent_dropout=0.3))(x)
    x = Bidirectional(LSTM(50, recurrent_dropout=0.3))(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu')(x)
    out = Dense(num_label, activation='softmax')(x)
    model = Model(inp, out)
    return model

app = Flask('D:\Me\minmaimin')
@app.route('/')
def show_predict_number_form():
    #return render_template(r'D:\Me\minmaimin\templates\predictorform')
    return render_template('predictorform.html')
@app.route('/result', methods=['POST'])
def result():
    form = request.form
    if request.method == 'POST':
      #write your function that loads the model
        max_len = 100
        word2vec = word_vector.get_model()
        word = request.form['word']
        model = tf.keras.models.load_model('minmaimin_LSTM.h5')
        #model = open('minmaimin_LSTM.h5', 'rb')
        predicted_number = process_query(word, model, word2vec, max_len)
        #predicted_number = model.predict(word)
        return render_template('resultsform.html', word=word, predicted_number=predicted_number)
app.run("localhost", "9999", debug=True)
print('OK')