from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocessor,model):
        self.preprocessor=preprocessor
        self.model=model

    def predict(self,X):
        try:
            X_transformed=self.preprocessor.transform(X)
            y_pred=self.model.predict(X_transformed)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def save_model(self,model_dir=SAVED_MODEL_DIR,model_file_name=MODEL_FILE_NAME):
        try:
            os.makedirs(model_dir,exist_ok=True)
            model_file_path=os.path.join(model_dir,model_file_name)
            with open(model_file_path,"wb") as f:
                import pickle
                pickle.dump(self.model,f)
        except Exception as e:
            raise NetworkSecurityException(e,sys)