import os
import sys
from dataclasses import dataclass

import spacy
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.downloader as api

from src.exception import CustomException
from src.logger import logging
from src.utils import preprocess_and_vectorize, remove_null_values, rename_columns, save_object,sentiment_mapper


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class TextPreprocessor:
    def __init__(self,nlp):
        self.nlp = nlp
    
    def fit(self, X, y=None):
        # This method is not used in this transformer
        return self
    
    def transform(self,X):
        print([preprocess_text(text, self.nlp) for text in X])
        return [preprocess_text(text, self.nlp) for text in X]
    
class TextVectorizer:
    def __init__(self,wv):
        self.wv = wv
    
    # def transform(self,X):
    #     return self.nlp(X).vector

    def fit(self, X, y=None):
        # This method is not used in this transformer
        return self

    def transform(self, X):
        vectorized = self.wv.get_mean_vector(X)
        print(vectorized)
        return np.stack(vectorized)

    
class SentimentMapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # The fit method is used for any setup that doesn't require access to the target variable
        return self

    def transform(self, X):
        modified_labels = [1 if label == 2 else 0 for label in X]
        return np.array(modified_labels).reshape(-1, 1)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.nlp = spacy.load("en_core_web_lg")
        self.wv = api.load("word2vec-google-news-300")

    def get_data_transformer_object(self):
        '''
        # this function is responsible for data transformation
        # '''
        # try:
        #     text_pipeline = Pipeline(
        #         steps = [
        #             ('preprocess',TextPreprocessor(nlp)),
        #             ('vectorizer',TextVectorizer(wv))
        #         ]
        #     )

        #     logging.info("Preprocessing and vectorization of text complete")

        #     preprocessor = ColumnTransformer(
        #         [
        #             # ('sentiment',SentimentMapper(), 'sentiment'),
        #             ('text',text_pipeline,'review_body')
        #         ]
        #     )
            
        #     logging.info("Sentiment mapping complete, pipeline implemented in transformer")
            
        #     return preprocessor
        
        # except Exception as e:
        #     raise CustomException(e,sys)

        return None
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data")

            train_df = train_df.rename(columns = {'0':'sentiment','1':'review_title','2':'review_body'}, inplace = False)
            test_df = test_df.rename(columns = {'0':'sentiment','1':'review_title','2':'review_body'}, inplace = False)

            logging.info("renamed columns in train and test dataset")

            train_df = remove_null_values(train_df)
            test_df = remove_null_values(test_df)
            print("Train DF shape : " ,train_df.shape)

            logging.info("removed null value rows from train and test data")


            logging.info("obtaining the preprocessing object")

            # preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "sentiment"
            other_columns = ["review_body","review_title"]

            input_feature_train_df = train_df[['review_body']]
            target_feature_train_df = train_df[[target_column_name]]
            print("input_feature_train df shape : ", input_feature_train_df.shape)
            print("target feature train df shape : ", target_feature_train_df.shape)

            print(input_feature_train_df.shape)


            input_feature_test_df = test_df[['review_body']]
            target_feature_test_df = test_df[[target_column_name]]

            logging.info("Applying preprocessing objcet on training and testing dataframe")

            feature_train_arrays = []
            indices_to_remove_train = []
            for i,text in enumerate(input_feature_train_df['review_body']):
                processed_array = preprocess_and_vectorize(text,self.nlp,self.wv)
                if processed_array is not None:
                    feature_train_arrays.append(processed_array)
                else:
                    indices_to_remove_train.append(i)
            input_feature_train_arr = np.vstack(feature_train_arrays)

            feature_test_arrays = []
            indices_to_remove_test = []
            for i,text in enumerate(input_feature_test_df['review_body']):
                processed_array = preprocess_and_vectorize(text, self.nlp, self.wv)
                if processed_array is not None:
                    feature_test_arrays.append(processed_array)
                else:
                    indices_to_remove_test.append(i)
            input_feature_test_arr = np.vstack(feature_test_arrays)

            # print(input_feature_train_arr[:2])
            # print(input_feature_train_arr[22])

            scaler = MinMaxScaler()
            input_feature_train_arr = scaler.fit_transform(input_feature_train_arr)
            
            input_feature_test_arr = scaler.transform(input_feature_test_arr)

            target_feature_train_df.loc[:, 'sentiment'] = target_feature_train_df['sentiment'].apply(sentiment_mapper)
            target_feature_test_df.loc[:,'sentiment'] = target_feature_test_df['sentiment'].apply(sentiment_mapper)

            target_feature_train_arr = target_feature_train_df.to_numpy()
            target_feature_train_arr = np.delete(target_feature_train_arr, indices_to_remove_train, axis=0)
            target_feature_train_arr = target_feature_train_arr.ravel()

            target_feature_test_arr = target_feature_test_df.to_numpy()
            target_feature_test_arr = np.delete(target_feature_test_arr, indices_to_remove_test, axis = 0)
            target_feature_test_arr = target_feature_test_arr.ravel()

            logging.info("Saved preprocessing object")

            # save_object(
            #     file_path = self.data_transformation_config.preprocessor_obj_file_path,
            #     obj = preprocessing_obj
            # )

            

            return(
                input_feature_train_arr,
                input_feature_test_arr,
                target_feature_train_arr,
                target_feature_test_arr
                # self.data_transformation_config.preprocessor_obj_file_path
            ) 

        except Exception as e:
            raise CustomException(e,sys)
