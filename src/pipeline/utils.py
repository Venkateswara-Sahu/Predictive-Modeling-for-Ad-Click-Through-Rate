import os
import sys
import numpy as np
import pandas as pd
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging

def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
        logging.info(f"Saved object to {file_path}")
            
    except Exception as e:
        logging.error(f"Error occurred while saving object to {file_path}")
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")
            
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
            
        logging.info(f"Loaded object from {file_path}")
        return obj
            
    except Exception as e:
        logging.error(f"Error occurred while loading object from {file_path}")
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models: dict, param: dict) -> dict:
    try:
        report = {}
        
        for model_name, model in models.items():
            # Get parameters for this model
            model_params = param.get(model_name, {})
            
            # Create and train model
            if model_params:
                gs = GridSearchCV(model, model_params, cv=3)
                gs.fit(x_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(x_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            # Calculate train and test scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            
            logging.info(f"{model_name} - Train Score: {train_score}, Test Score: {test_score}")
            report[model_name] = test_score
            
        return report
            
    except Exception as e:
        logging.error("Error occurred while evaluating models")
        raise CustomException(e, sys)
