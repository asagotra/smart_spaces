## Data Model to fecth the data from thingsboard APi and store the data in postgreSQL

# import libraries 

import requests
import json
import datetime
import numpy as np
import pandas as pd
import sqlalchemy  # Package for accessing SQL databases via Python
import time


# DataFrame to initalize
data_export=pd.DataFrame({'TimeStamp':0, 'temperature': 0, 
                            'humidity': 0, 'headCount':0, 
                            'DoorStatus': 'close', 
                            'WindowStatus': 'open'}, 
                            index=[0])


def extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    return results

# Function to create the X-Autherization Token

def getToken():

    '''
    Please provide your local thingsboard username and password credentials

    '''
    url = 'http://localhost:8080' + '/api/auth/login'
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    loginJSON = {'username': 'tenant@thingsboard.org', 'password': 'your password'} 
    tokenAuthResp = requests.post(url, headers=headers, json=loginJSON).json()
    token = tokenAuthResp['token']
    return token

token = getToken()


while True:
    # send requests to the api 
    # 
    # Data Extract from thingsboard api

    deviceId='your device id' # Write your own device_Id
    url_base = f'http://localhost:8080/api/plugins/telemetry/DEVICE/{deviceId}/values/timeseries?keys'
    headers = {'X-Authorization':'Bearer ' + token}
    r = requests.get(url_base, headers=headers) 
    data= r.json()


    # Create the dictionary 

    column_names = ['TimeStamp',
                    'headCount', 
                    'q1_headcount', 
                    'q2_headcount', 
                    'q3_headcount', 
                    'q4_headcount', 
                    'appliances',
                    'temperature', 
                    'humidity', 
                    'doorstatus', 
                    'windowstatus',
                    'thermalload',
                    'q1_thermalload',
                    'q2_thermalload',
                    'q3_thermalload',
                    'q4_thermalload',
                    'loadforecast', 
                    'energyconsumption',
                    'energysaving', 
                    'summary', 
                    'icon', 
                    'precipintensity', 
                    'precipprobability', 
                    'externaltemperature',
                    'apparenttemperature', 
                    'dewpoint', 
                    'externalhumidity', 
                    'externalpressure', 
                    'windspeed', 
                    'windgust', 
                    'windbearing', 
                    'CloudCover',
                    'uvindex', 
                    'visibility', 
                    'ozone', 
                    'q1_temperature', 
                    'q2_temperature',
                    'q3_temperature',
                    'q4_temperature',
                    'q1_appliances',
                    'q2_appliances',
                    'q3_appliances',
                    'q4_appliances']
    column_datatypes = ['Date',
                        'integer', 
                        'integer',
                        'integer',
                        'integer', 
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'string',
                        'string',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'string',
                        'string',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer',
                        'integer']

    schema_dict = dict(zip(column_names, column_datatypes))
    #print(schema_dict)

    # convert unix timestamp to pandas datetime index
    Time = extract_values(data['temperature'],'ts')

    # Data store in the dictionary
    schema_dict['TimeStamp'] = pd.to_datetime(Time, unit='ms')
    #schema_dict['TimeStamp'] = 
    schema_dict['headCount'] = extract_values(data['headCount'],'value')
    schema_dict['q1_headcount'] = extract_values(data['q1_headcount'],'value')
    schema_dict['q2_headcount'] = extract_values(data['q2_headcount'],'value')
    schema_dict['q3_headcount'] = extract_values(data['q3_headcount'],'value')
    schema_dict['q4_headcount'] = extract_values(data['q4_headcount'],'value') 
    schema_dict['appliances'] = extract_values(data['appliances'],'value') 
    schema_dict['temperature'] =  extract_values(data['temperature'],'value')
    schema_dict['humidity'] = extract_values(data['humidity'],'value')
    schema_dict['doorstatus'] = extract_values(data['doorstatus'],'value')
    schema_dict['windowstatus'] = extract_values(data['windowstatus'],'value')
    schema_dict['thermalload'] = extract_values(data['thermalload'],'value')
    schema_dict['q1_thermalload'] = extract_values(data['q1_thermalload'],'value')
    schema_dict['q2_thermalload'] = extract_values(data['q2_thermalload'],'value')
    schema_dict['q3_thermalload'] = extract_values(data['q3_thermalload'],'value')
    schema_dict['q4_thermalload'] = extract_values(data['q4_thermalload'],'value')
    schema_dict['loadforecast'] = extract_values(data['loadforecast'],'value')
    schema_dict['energyconsumption'] = extract_values(data['energyconsumption'],'value')
    schema_dict['energysaving'] = extract_values(data['energysaving'],'value')
    schema_dict['summary'] = extract_values(data['summary'],'value')
    schema_dict['icon'] = extract_values(data['icon'],'value')
    schema_dict['precipintensity'] = extract_values(data['precipintensity'],'value')
    schema_dict['precipprobability'] = extract_values(data['precipprobability'],'value')
    schema_dict['externaltemperature'] = extract_values(data['externaltemperature'],'value')
    schema_dict['apparenttemperature'] =  extract_values(data['apparenttemperature'],'value')
    schema_dict['dewpoint'] = extract_values(data['dewpoint'],'value')
    schema_dict['externalhumidity'] = extract_values(data['externalhumidity'],'value')
    schema_dict['externalpressure'] = extract_values(data['externalpressure'],'value')
    schema_dict['windspeed'] = extract_values(data['windspeed'],'value')
    schema_dict['windgust'] = extract_values(data['windgust'],'value')
    schema_dict['windbearing'] = extract_values(data['windbearing'],'value')
    schema_dict['cloudcover'] = extract_values(data['cloudcover'],'value')
    schema_dict['uvindex'] = extract_values(data['uvindex'],'value')
    schema_dict['visibility'] = extract_values(data['visibility'],'value')
    schema_dict['ozone'] = extract_values(data['ozone'],'value')
    schema_dict['q1_temperature'] = extract_values(data['q1_temperature'],'value')
    schema_dict['q2_temperature'] = extract_values(data['q2_temperature'],'value')
    schema_dict['q3_temperature'] = extract_values(data['q3_temperature'],'value')
    schema_dict['q4_temperature'] = extract_values(data['q4_temperature'],'value')
    schema_dict['q1_appliances'] = extract_values(data['q1_appliances'],'value')
    schema_dict['q2_appliances'] = extract_values(data['q2_appliances'],'value')
    schema_dict['q3_appliances'] = extract_values(data['q3_appliances'],'value')
    schema_dict['q4_appliances'] = extract_values(data['q4_appliances'],'value')


    # check if same timestamp is available or not and convert data into pandas dataframe 
    if schema_dict['TimeStamp'] != data_export['TimeStamp'][0]:
        data_export = pd.DataFrame.from_dict(schema_dict)

    # Pushing dataframe to PostgresSQL Database
    # Connect to database (Note: The package psychopg2 is required for Postgres to work with SQLAlchemy)
    # engine = sqlalchemy.create_engine("postgresql://username:password@localhost/database_name")
    # Provide your username, password and database credentials

        engine = sqlalchemy.create_engine("postgresql://your_username:your_password@localhost/your_database_name")
        con = engine.connect()
        
    # Verify that there are no existing tables
        print(engine.table_names())

    # create table name where you store the data
        table_name = 'BEMS'
        data_export.to_sql(table_name, con, if_exists='append')

        con.close()
        
    #  query to fetch data from PostgresSQL
    '''
    SELECT * FROM public.BEMS
    '''
    time.sleep(300) # sleep for 300 seconds
   






