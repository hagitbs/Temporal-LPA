
import requests
import json
from requests_oauthlib import OAuth1
import pandas as pd 

table_id = '' 
project_id = ''
#pandas_gbq.to_gbq(ns_results, table_id, project_id=project_id)

 
from google.oauth2 import service_account
import pandas_gbq

 
credentials = service_account.Credentials.from_service_account_info(
{
 
}) 
# df = pandas_gbq.read_gbq(sql, project_id="YOUR-PROJECT-ID", credentials=credentials)   

