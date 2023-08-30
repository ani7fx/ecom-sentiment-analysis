import spacy
import pandas as pd
import sys
import os
import numpy as np
import dill
from src.exception import CustomException
import gensim.downloader as api
from sklearn.metrics import accuracy_score



wv = api.load("word2vec-google-news-300")

def preprocess_and_vectorize(text,nlp,wv):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
        
    
    if not filtered_tokens:
        return None
    return np.array(wv.get_mean_vector(filtered_tokens))


def rename_columns(df):
    df.rename(columns={0: 'sentiment', 1: 'review_title', 2: 'review_body'}, inplace=True)
    return df

def remove_null_values(df):
    df.dropna(subset = ['review_title'], inplace = True)
    return df

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def sentiment_mapper(x):
    if x == 2:
        return 1
    else:
        return 0
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_train_pred =[1 if value > 0.5 else 0 for value in y_train_pred]

            y_test_pred = model.predict(X_test)
            y_test_pred =[1 if value > 0.5 else 0 for value in y_test_pred]

            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_score

        return report


    except Exception as e:
        raise CustomException(e,sys)





