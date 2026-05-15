from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import GridSearchCV
import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import sys
import os
import numpy as np
import pandas as pd
import dill
import pickle

from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
    
def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
                yaml.dump(content,file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_numpy_array_data(file_path:str,array:np.array)->None:
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_object(file_path:str,obj:object)->None:
    try:
        logging.info(f"Entered the save_object method of MainUtils class")
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
        logging.info(f"Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_object(file_path:str)->object:
    try:
        logging.info(f"Entered the load_object method of MainUtils class")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file: {file_path} is not found")
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_numpy_array_data(file_path:str)->np.array:
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models:dict,param:dict)->dict:
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            logging.info(f"Hyperparameter tuning for {list(models.keys())[i]}")
            gs=GridSearchCV(model,para,cv=5,verbose=1,n_jobs=-1)
            gs.fit(X_train,y_train)
            logging.info(f"Best params for {list(models.keys())[i]}: {gs.best_params_}")
            
            # Use the best estimator from GridSearchCV
            best_model=gs.best_estimator_
            
            y_train_pred=best_model.predict(X_train)
            y_test_pred=best_model.predict(X_test)
            
            # Use F1-score for classification instead of r2_score
            train_model_score=f1_score(y_true=y_train,y_pred=y_train_pred)  
            test_model_score=f1_score(y_true=y_test,y_pred=y_test_pred)
            
            logging.info(f"Train F1 Score: {train_model_score}, Test F1 Score: {test_model_score}")
            report[list(models.keys())[i]]=test_model_score
        return report
    except Exception as e:
        raise NetworkSecurityException(e,sys)