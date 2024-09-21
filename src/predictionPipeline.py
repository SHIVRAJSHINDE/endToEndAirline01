import sys
import pandas as pd
from src.exp.utils import load_objects
from src.component.st01preProcessingTrainingFunctions import streamlineData
from src.component.st03preProcessingPredtionFunction import predctionFunctions
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
            predction_Functions = predctionFunctions()

            df = predction_Functions.extract_Deptdate_time(df)   
            df = predction_Functions.extract_Arrdate_time(df)

            df = streamline_data.extract_date_components(df)
            
            df = predction_Functions.calculateDuration(df)
            
            df = streamline_data.extract_Dept_Hrs_Minutes(df,'Dep_Time')
            df = streamline_data.extract_Arr_Hrs_Minutes(df,'Arrival_Time')
            columns_to_drop = ["Arrival_Time","Arrival_Time","Dep_Time","Date_of_Journey","Duration"]
            df = streamline_data.drop_unnecessary_columns(df,columns_to_drop)
            df = predction_Functions.restructure_columns_predictionPipeline(df)
            print(df.T)
            df = predction_Functions.preprocesing(df)
            ouput = predction_Functions.predict(df)
            print(df.T)

            return ouput
        
        