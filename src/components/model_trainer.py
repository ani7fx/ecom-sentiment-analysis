import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,input_train_array,input_test_array,target_train_array,target_test_array):
        try:
            logging.info("Splitting training and test input")

            X_train, y_train, X_test, y_test = (
                input_train_array,
                target_train_array,
                input_test_array,
                target_test_array
            )

            print("X_train shape : ",X_train.shape)
            print("y_train shape :", y_train.shape)

            models = {
                "Naive Bayes" : MultinomialNB(),
                "Gradient Boosting" : GradientBoostingClassifier(),
                "Random Forest Classifier" : RandomForestClassifier(),
                "CatBoost Classifier" : CatBoostClassifier(verbose = False),
                "XGBoost Classifier" : XGBClassifier(),
                "Ridge" : Ridge(),
                "Ada Boost Regressor" : AdaBoostClassifier(),
                "KNeighbors Classifier" : KNeighborsClassifier()
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train,
                                               X_test = X_test, y_test = y_test, models = models)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            best_report = accuracy_score(y_test, predicted)
            print("The best model was :", best_model_name, " with an accuracy of")
            return best_report



        except Exception as e:
            raise CustomException(e,sys)
