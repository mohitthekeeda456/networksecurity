import os
import sys
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation


from networksecurity.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from networksecurity.entity.artifact_entity import(
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config=TrainingPipelineConfig()
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            logging.info("starting data ingestion")
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            logging.info("starting data validation")
            self.data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation=DataValidation(data_validation_config=self.data_validation_config,data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact=data_validation.initiate_data_validation()
            logging.info(f"data validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            logging.info("starting data transformation")
            self.data_transformation_config=DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation=DataTransformation(data_validation_artifact=data_validation_artifact,Data_transformation_config=self.data_transformation_config)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info(f"data transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            logging.info("starting model trainer")
            self.model_trainer_config=ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer=ModelTrainer(model_trainer_config=self.model_trainer_config,data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact=model_trainer.initiate_model_trainer()
            logging.info(f"model trainer completed and artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)