import logging
import os
import sys
import pandas as pd 
from src.pipeline.exception import CustomException 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s'
)

logging.basicConfig(level=logging.INFO)
@dataclass
class DataIngestionconfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionconfig()
        self.data_path = 'notebook/stud.csv'
    def initiate_data_ingestion(self) -> pd.DataFrame:
        logging.info("Entered the data ingestion method")
        
        try:
            df = pd.read_csv(self.data_path)
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )
        
        except Exception as e:
            logging.error("Exception occurred during data ingestion")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        # Initialize data ingestion
        obj = DataIngestion()
        
        # Start data ingestion
        print("\n1. Starting data ingestion process...")
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        print("✓ Data ingestion completed successfully!")
        
        # Proceed with data transformation
        print("\n2. Starting data transformation process...")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        print("✓ Data transformation completed successfully!")
        print(f"  • Transformed arrays shapes: Train-{train_arr.shape}, Test-{test_arr.shape}")
        print(f"  • Preprocessor saved at: {preprocessor_path}")
        
        # Start model training
        print("\n3. Starting model training process...")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print("✓ Model training completed successfully!")
        print(f"  • Best model R² score: {r2_score:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        raise e