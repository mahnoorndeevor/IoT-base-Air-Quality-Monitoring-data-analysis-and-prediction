import requests
import pyodbc
import time
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar




# Function to fetch and preprocess data
def fetch_and_preprocess_data(api_url):
    
    response = requests.get(api_url)
    data = response.json()

    # Assuming the relevant data is under the 'feeds' key
    if 'feeds' in data:
        df = pd.DataFrame(data['feeds'])  # Create DataFrame from 'feeds' list
    else:
        print("Error: 'feeds' key not found in the response.")
        return None
    # Example preprocessing - rename columns as needed
    #columnsToDrop=['latitude', 'longitude', 'elevation', 'status', 'entry_id']
    
    #df=df.drop(columnsToDrop, axis=1)
    #df['created_at'] = pd.to_datetime(df['created_at'])
    
    #df['date_'] = df['created_at'].dt.date
    #df['time_'] = df['created_at'].dt.time

    #df=df.drop('created_at', axis=1)
    df= df.rename(columns = {'field1':'CO'})
    df= df.rename(columns = {'field2':'CO2'})
    df= df.rename(columns = {'field3':'Methane'})
    df= df.rename(columns = {'field4':'PM2_5'})
    df= df.rename(columns = {'field5':'PM10'})
    df= df.rename(columns = {'field6':'Temperature'})
    df= df.rename(columns = {'field7':'Humidity'})
    


    return df

# Function to insert data into SQL Server
def insert_data_to_sql(df, cnxn_str, table_name):
    cnxn = pyodbc.connect(cnxn_str)
    cursor = cnxn.cursor()

    for index, row in df.iterrows():
        query = f"INSERT INTO {table_name} (CO, CO2, Methane, [PM2_5], PM10, Temperature, Humidity) VALUES (?, ?, ?, ?, ?, ?, ?)"
        values = (row['CO'], row['CO2'], row['Methane'], row['PM2_5'], row['PM10'], row['Temperature'], row['Humidity'])
        cursor.execute(query, values)

    cnxn.commit()
    cursor.close()
    cnxn.close()

# Main loop
def main():
    api_url = "https://api.thingspeak.com/channels/2484481/feeds.json?api_key=90PG7TU0B8EO06K1&results=2"
    server = 'DESKTOP-TS7G0Q5\SQLEXPRESS'
    database = 'AQInode'
    table_name = 'testable'
    cnxn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes'

    while True:
        df = fetch_and_preprocess_data(api_url)
        insert_data_to_sql(df, cnxn_str, table_name)
        print("Data fetched, processed, and inserted into SQL Server successfully.")
        time.sleep(20)  # Polling interval: Adjust as needed

if __name__ == "__main__":
    main() 
