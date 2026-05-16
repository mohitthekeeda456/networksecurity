import sys
import os
import certifi
from flask import request

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME
from networksecurity.utils.ml_utils import model
from networksecurity.utils.ml_utils import model
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
ca=certifi.where()
from dotenv import load_dotenv
load_dotenv()
mongo_db_url=os.getenv("MONGO_DB_URL")
print(mongo_db_url)
import pymongo
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

client=pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)
database=client[DATA_INGESTION_DATABASE_NAME]
colleection=database[DATA_INGESTION_COLLECTION_NAME]
app=FastAPI()
origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
from fastapi.templating import Jinja2Templates
templates=Jinja2Templates(directory="./templates")


@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/train",tags=["training"])
async def train_route(request:Request):
    try:
        logging.info("request received for training")
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("training is successful")
    except Exception as e:
        raise NetworkSecurityException(e,sys)

@app.post("/predict")
async def predict_route(request:Request,file:UploadFile=File(...)):
    try:
        logging.info("request received for prediction")
        
        # Create output directory if it doesn't exist
        os.makedirs("predicted_output", exist_ok=True)
        
        df=pd.read_csv(file.file)
        logging.info("CSV file read successfully")
        
        preprocessor=load_object("final_model/preprocessor.pkl")
        logging.info("Preprocessor loaded successfully")
        
        final_model=load_object("final_model/model.pkl")
        logging.info("Model loaded successfully")
        
        network_model=NetworkModel(preprocessor=preprocessor,model=final_model)
        print(df.iloc[0])
        y_pred=network_model.predict(df)
        print(y_pred)
        df["predicted_column"]=y_pred
        print(df["predicted_column"])
        df.to_csv("predicted_output/output.csv")
        logging.info("Predictions saved to predicted_output/output.csv")
        
        table_html=df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html",{"request":request,"table":table_html})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise NetworkSecurityException(e,sys)



    
if __name__=="__main__":
    app_run(app,host="localhost",port=8000)