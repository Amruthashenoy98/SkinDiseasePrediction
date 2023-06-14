import os
import sys
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            models = {
                "Decision tree":DecisionTreeClassifier(random_state=42),
                "KNN": KNeighborsClassifier(),
                "Random forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "CatBoost": CatBoostClassifier(random_seed=42,verbose=False),
                "Xgb classifier": XGBClassifier(),
    
            }
            params={
            
                "KNN":{
                    'n_neighbors':[3,5,7],
                    'weights':['uniform','distance'],
                    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Decision tree":{
                    'criterion': ["gini", "entropy"],
                    'splitter':['best', 'random'],
                    'max_depth':[2,3,4,5],
                    'min_samples_leaf': [5, 10, 20, 50, 10]
                },
                "Random forest":{
                    'n_estimators': [50,100],
                    'max_features': ['auto', 'sqrt'],
                    'max_depth': [2,3,4,5],
                    'min_samples_split': [7, 10]
                },
                "Gradient Boosting":{
                    "n_estimators":[5,50,100],
                    "max_depth":[5,7],
                    "learning_rate":[0.01,0.1,1,10,100]
                },
                "AdaBoost":{
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.1, 0.5, 1.0]
                },
                "CatBoost":{
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.1, 0.5, 1.0]
                },
                "Xgb classifier":{
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [100, 500]
                }
                
                }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            f1score = f1_score(y_test, predicted,average='weighted')
            return best_model_name,f1score
            



            
        except Exception as e:
            raise CustomException(e,sys)