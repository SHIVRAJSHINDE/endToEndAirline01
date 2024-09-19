import sys
import pandas as pd
from src.exp.utils import load_objects
from src.component.st01apreprocessingFunctions import streamlineData
class PredictPipeline():
    def __init__(self):
        pass

    def predict(self,features):
        modelPath = "artifacts/model.pkl"
        preprocessorPath = "artifacts/preprocessor.pkl"
        model = load_objects(file_path=modelPath)
        transformation = load_objects(file_path=preprocessorPath)
        dataScaled = transformation.transform(features)
        pred = model.predict(dataScaled)
        return pred
    



class CustomData():

        def receiveDataFromWeb(self,Airline:str,Date_of_Journey:str,Source:str,Destination:str,
                     Dep_Time:str,Arrival_Time:str,Duration:str,Total_Stops:str):
            inputDict = {
                "Airline": [Airline],
                "Date_of_Journey": [Date_of_Journey],
                "Source": [Source],
                "Destination": [Destination],
                "Dep_Time": [Dep_Time],
                "Arrival_Time": [Arrival_Time],
                "Duration": [Duration],
                "Total_Stops": [Total_Stops]

            }
            df = pd.DataFrame(inputDict)
            print(df)

            streamline_data = streamlineData()
            df = streamline_data.extract_Deptdate_time(df)   
            df = streamline_data.extract_Arrdate_time(df)         
            df = streamline_data.extract_date_components(df)
            df = streamline_data.calculateDuration(df)
            df = streamline_data.extract_Dept_Hrs_Minutes(df,'Dep_Time')
            df = streamline_data.extract_Arr_Hrs_Minutes(df,'Arrival_Time')
            columns_to_drop = ["Arrival_Time","Arrival_Time","Dep_Time","Date_of_Journey","Duration"]
            df = streamline_data.drop_unnecessary_columns(df,columns_to_drop)
            df = streamline_data.restructure_columns_predictionPipeline(df)
            print(df.T)
            df = streamline_data.preprocesing(df)
            ouput = streamline_data.predict(df)
            print(df.T)

            return ouput
        
        