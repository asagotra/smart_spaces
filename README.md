# Intial Setup 

## install the required python libaries

pip install -r requirements.txt

## Raspberry Pi and ThingsBoard

After setting up the raspberry pi and integrating it with the temperature sensors and the camera 
you need to send the data (i.e. temperature, humidity, people count, appliances count)from the rasberryPi 
to the Thingsboard IoT platform.

For that, on Thingsboard, device is the place where the actual data comes from the raspberrPi device. 
So, after creating a device on Thingsboard, you need to replace the access token in the raspberryPi.py file 
with the access token of the device you created

## Dark Sky for external parameters

In the dark_sky.py python file please provide your own secret key by 
logging into darksy API with your own email ID and password.

## Data Model(Fetch data from IoT platform and stored in the postgreSQL)

To run the data_model.py script,  first you need to create X - Authorization token.
For this, you need to provide your Thingsboard username and password credentials in the data_model python file (see line 48).
Secondly to send request to ThingsBoard api and retreive data from Thingsboard you need to provide the 
device id in data_model.py file (see line 60) of device where you pushed the data.
Finally, when you save the data to the local postgreSQL provide your postgreSQL username, password and database name (see line 206).


# People and appliances count model and data push to the ThingsBoard

## Run the python script

python raspberryPi.py connect


# Fetch data from ThingsBoard and Stored in the PostgreSQL Database

## Run the python script

python data_model.py

Run both python script parallel

All the python scripts are also well commented for the user to understand it.


# Installation of Local ThingsBoard community edition

## Raspberry Pi

### Please follow this link

https://thingsboard.io/docs/user-guide/install/rpi/

## Windows

https://thingsboard.io/docs/user-guide/install/windows/

## Ubuntu

https://thingsboard.io/docs/user-guide/install/ubuntu/

