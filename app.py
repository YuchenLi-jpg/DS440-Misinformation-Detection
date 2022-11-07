import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("tensorflow")
import tensorflow
from tensorflow.keras import layers
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import tensorflow.keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import string

Navigation = {"page_title":"MisinformationDetection.io","page_icon":"🤥","layout":"centered"}
st.set_page_config(**Navigation)


def load_model():
    with open("birectionalISTM_model.pkl", "rb") as file:
        data = pickle.load(file)    
    return data


def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in str(string.punctuation)])
    return punctuationfree

model = load_model()
tokenizer = load_tokenizer()

def main():
    st.header("Misinformation Detection")

    st.write("""### Please enter your text:""")

    # construct text box
    text = [st.text_input('')]

    result = st.button('Classify')
    if result:
        if text:
            text = [remove_punctuation(text[0]).lower()]
            processed = tokenizer.texts_to_sequences(text)
            processed = pad_sequences(processed, maxlen=1000)
            result = model.predict(processed)[0].astype(float)[0]
            prediction = (model.predict(processed)>= 0.5).astype(int)
            if prediction == [[0]]:
              st.subheader(f"The article is fact")
            else:
              st.subheader(f"The article is fake")
              
            chart_data = pd.DataFrame(
            [[1-result, result]],
            columns=["True", "Fake"])

            st.bar_chart(chart_data)
    
       


if __name__ == '__main__':
    main()
