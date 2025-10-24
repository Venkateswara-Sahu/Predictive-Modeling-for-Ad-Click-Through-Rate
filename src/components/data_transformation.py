import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object, load_object

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationconfig()
    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and categorical pipelines created")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            logging.error("Exception occurred in data transformation object creation")
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data for transformation")
            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = 'math_score'
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing data")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessor object")
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        except Exception as e:
            logging.error("Exception occurred during data transformation")
            raise CustomException(e, sys)
if __name__ == "__main__":
    data_transformation = DataTransformation()
    train_path = 'artifacts/train.csv'
    test_path = 'artifacts/test.csv'
    data_transformation.initiate_data_transformation(train_path, test_path)