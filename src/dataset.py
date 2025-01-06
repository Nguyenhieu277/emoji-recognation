import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import string

df_train = pd.read_csv("dataset/train.txt", delimiter = ';', header = None, names = ["text", "label"])

df_val = pd.read_csv("dataset/val.txt", delimiter = ';', header = None, names = ["text", "label"])

df_test = pd.read_csv("dataset/test.txt", delimiter = ';', header = None, names = ["text", "label"])

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    cleaned_text = ' '.join(tokens)
    return cleaned_text

def clean_dataframe(df):
    df.drop_duplicates()
    df = df[df["label"] != "love"]
    df = df[df["label"] != "surprise"]
    return df

df_train["cleaned_text"] = df_train["text"].apply(clean_text)
df_val["cleaned_text"] = df_train["text"].apply(clean_text)
df_test["cleaned_text"] = df_test["text"].apply(clean_text)

df_train.to_csv("dataset/cleaned/train.csv", index = False)
df_val.to_csv("dataset/cleaned/val.csv", index = False)
df_test.to_csv("dataset/cleaned/test.csv", index = False)