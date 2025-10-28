import sys
import os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def initiate_pipeline(self):
        try:
            logging.info("Starting data ingestion phase...")
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")

            logging.info("Starting data transformation phase...")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(train_path, test_path)
            logging.info(f"Data transformation completed. Train shape: {train_arr.shape}, Test shape: {test_arr.shape}")

            logging.info("Starting model training phase...")
            results = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info("Model training completed")

            # Print top 2 models and their performance
            logging.info("\nTop 2 Models Performance:")
            for model_name, score in results["top_2_models"].items():
                logging.info(f"{model_name}: R² Score = {score:.4f}")

            logging.info(f"\nBest Model Selected: {results['best_model_name']}")
            logging.info(f"Best Model Final R² Score: {results['best_model_score']:.4f}")

            return results

        except Exception as e:
            logging.error(f"Pipeline failed with error: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            import traceback
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Ensure artifacts directory exists
        os.makedirs("artifacts", exist_ok=True)
        
        print("\nStarting Training Pipeline...")
        pipeline = TrainPipeline()
        results = pipeline.initiate_pipeline()
        
        print("\nTraining Pipeline completed successfully!")
        print("\nTop 2 Models Performance:")
        for model_name, score in results["top_2_models"].items():
            print(f"• {model_name}: R² Score = {score:.4f}")
        
        print(f"\nBest Model Selected: {results['best_model_name']}")
        print(f"Best Model Final R² Score: {results['best_model_score']:.4f}")
        
    except Exception as e:
        print(f"\nError occurred during pipeline execution:")
        print(f"Type: {type(e).__name__}")
        print(f"Details: {str(e)}")
        raise
