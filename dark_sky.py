import requests
import json

def fetch_darksky():
    """
    Login to darksky api wiht gmail account and password
    you will get the secret key.
    
    """
    secret_key = 'Your Secret Key' # Provide your own secret key
    
    """
    These longitude and latitude of Banglore. 
    But you need to provide these deatails by location wise.
    
    """
    longitude = 77.594563
    latitude = 12.971599
    exclude = 'minutely, hourly, daily, alerts, flags'

    url_darksky = f'https://api.darksky.net/forecast/{secret_key}/{latitude},{longitude}?units=si&exclude={exclude}'

    response = requests.get(url_darksky)
    response_json = response.json()

    return response_json