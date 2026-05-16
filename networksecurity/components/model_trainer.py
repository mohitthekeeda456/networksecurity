import os
import sys
from sklearn.model_selection import GridSearchCV
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_object,load_object,load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

import dagshub
dagshub.init(repo_owner='mohitthekeeda456', repo_name='networksecurity', mlflow=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import mlflow                                       





class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score
            
            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")


    def train_model(self,X_train,y_train,X_test,y_test):
        models={
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }
        params={
            "Logistic Regression": {
                'C': [0.1, 1, 10],
                'solver': ['liblinear']
            },
            "KNN": {
                'n_neighbors': [3, 5, 7]
            },
            "Decision Tree": {
                'max_depth': [3, 5, 7]
            },
            "Random Forest": {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7]
            },
            "AdaBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.1, 0.5, 1.0]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.1, 0.5, 1.0]
            }
        }
        logging.info(f"Training and tuning {len(models)} models with hyperparameter tuning...")
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
        logging.info(f"Model Report: {model_report}")
        best_model_score=max(sorted(model_report.values()))
        best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        logging.info(f"Best Model: {best_model_name} with Test F1 Score: {best_model_score}")
        
        # Retrain the best model with its optimal parameters
        best_model=models[best_model_name]
        best_params=params[best_model_name]
        gs=GridSearchCV(best_model,best_params,cv=5,verbose=1,n_jobs=-1)
        gs.fit(X_train,y_train)
        best_model=gs.best_estimator_
        
        y_train_pred=best_model.predict(X_train)
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        #track ml flow
        self.track_mlflow(best_model,classification_train_metric)
        
        y_test_pred=best_model.predict(X_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
        
        self.track_mlflow(best_model,classification_test_metric)
        
        Preprocessor=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)
        Network_Model=NetworkModel(preprocessor=Preprocessor,model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=Network_Model)
        
        save_object("final_model/model.pkl",best_model)
        
        
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                  train_metric_artifact=classification_train_metric,
                                                  test_metric_artifact=classification_test_metric,)
        
        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        return model_trainer_artifact




    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path
            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)
            x_train,y_train=train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test=test_arr[:,:-1],test_arr[:,-1]
            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)