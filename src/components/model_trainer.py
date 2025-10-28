import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


from src.pipeline.exception import CustomException
from src.pipeline.logger import logging


from src.pipeline.utils import save_object,evaluate_models


#in every file we have to make config class where we can define all the input requires


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]


            )
            models={
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor(),
                "Gradient Boosting":GradientBoostingRegressor()
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
               
            }


            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)  #this function we have creayted in utils


            # Sort models by score and get top 2
            sorted_models = dict(sorted(model_report.items(), key=lambda x: x[1], reverse=True))
            top_2_models = dict(list(sorted_models.items())[:2])
            
            # Get the best model
            best_model_name = list(sorted_models.keys())[0]
            best_model = models[best_model_name]
            best_model_score = sorted_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found (all models performed poorly with R² < 0.6)")
            
            logging.info(f"Best models found on training and testing dataset:")
            for model_name, score in top_2_models.items():
                logging.info(f"{model_name}: R² = {score:.4f}")
           
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Final predictions with best model
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            
            return {
                "best_model_name": best_model_name,
                "best_model_score": r2_square,
                "top_2_models": top_2_models
            }


        except Exception as e:
            raise CustomException(e,sys)


