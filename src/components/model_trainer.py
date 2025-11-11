import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params={
                "Decision Tree": {
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 10]
                    },
                "Random Forest":{
                    'n_estimators': [100, 200],
                    'max_depth': [20, None]
                    },
                "Gradient Boosting":{
                    'learning_rate':[0.05, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7]
                    },
                "Linear Regression":{},
                "K-Neighbors Regressor":{
                    'n_neighbors': [5, 7]
                    },
                "XGBRegressor":{
                    'learning_rate':[0.05, 0.1],
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7]
                    },
                "CatBoosting Regressor":{
                    'depth': [6, 8],
                    'learning_rate': [0.05, 0.1],
                    'iterations': [100, 200]
                    },
                "AdaBoost Regressor":{
                    'learning_rate':[0.1, 0.5],
                    'n_estimators': [50, 100]
                    }
                    }
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)] 
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2_sqaure=r2_score(y_test,predicted)

            return r2_sqaure
        except Exception as e:
            raise CustomException(e,sys)