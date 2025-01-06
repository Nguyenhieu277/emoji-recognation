from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from model import CNN

df_train = pd.read_csv("dataset/cleaned/train.csv")
df_test = pd.read_csv("dataset/cleaned/test.csv")
df_val = pd.read_csv("dataset/cleaned/val.csv")

def convert_df_to_X_y(df):
    X = []
    for text in df["cleaned_text"]:
        X.append(text)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    y = to_categorical(y)

    return np.array(X), y

def text_to_embedding(text):
    tokenizer = Tokenizer(num_words = 1000)
    tokenizer.fit_on_texts(text)
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, maxlen = 100)
    return np.array(X)

X_train, y_train = convert_df_to_X_y(df_train)
X_test, y_test = convert_df_to_X_y(df_test)
X_val, y_val = convert_df_to_X_y(df_val)

X_train = text_to_embedding(X_train)
X_test = text_to_embedding(X_test)
X_val = text_to_embedding(X_val)

model = CNN(1000, 100)
model.compile()
model.fit(X_train, y_train, X_val, y_val)
model.summary()
model.save("model_CNN.h5")



