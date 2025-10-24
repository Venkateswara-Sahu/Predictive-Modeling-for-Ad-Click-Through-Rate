import os
import sys
import pandas as pd
import numpy as np
import pickle
import dill
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
