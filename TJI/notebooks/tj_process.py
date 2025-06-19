import pandas as pd
import numpy as np
from google.cloud import bigquery
# Initialize BigQuery client
client = bigquery.Client()

# -------------------------------
# Fetching transactional data
# -------------------------------
transactional_query = """
SELECT 
        StoreId,
        UniqueCheckId,
        PLU,
        Date,
        CheckTime,
FROM `zeta-scene-450822-n1.tj.store_ticket_sales_by_day'
Limit 10
"""
# Run the query and load data into a DataFrame
transctional_df = client.query(transactional_query).to_dataframe()

# ------------------------------- -------------------------------
# Preprocessing transactional data
# ------------------------------- -------------------------------

transctional_df['date']=transctional_df['Date'].dt.date
transctional_df['time'] = transctional_df['CheckTime'].dt.time


# Concatenate 'StoreID', 'Date', and 'CheckTime' to create 'Key'
transctional_df['Key'] = transctional_df['StoreId'].astype(str) + "_" + transctional_df['date'].astype(str) + "_" + transctional_df['time'].astype(str)



# ------------------------------- -------------------------------
# Making summary table of  transactional data
# ------------------------------- -------------------------------
summary_trans=transctional_df.groupby(['Key', 'UniqueCheckId'])['PLU'].unique().reset_index()
print(summary_trans.head())




# ------------------------------- -------------------------------
# Fetching loyalty data
# ------------------------------- -------------------------------




# LOYALTY PREPROCESS
loyalty_query = """
SELECT 
        Receipt DateTime,
        Menu Item ID,
        Checkin ID,
     
FROM `zeta-scene-450822-n1.tj.store_ticket_sales_by_day'
Limit 10
"""
# Run the query and load data into a DataFrame
loyalty_df = client.query(loyalty_query).to_dataframe()


float_col=['Menu Item ID','Checkin ID']

loyalty_df['Date'] = loyalty_df['Date'].apply(pd.to_datetime, errors='coerce')
loyalty_df[float_col] =loyalty_df[float_col] .apply(pd.to_numeric, errors='coerce')

loyalty_df['date']=loyalty_df['Receipt DateTime'].dt.date

loyalty_df['StoreId'] = loyalty_df['Location Name'].str[-4:].astype(float)
# loyalty_df['StoreId']=loyalty_df['StoreId'].astype(float)

loyalty_df['time'] = np.where(loyalty_df['StoreId'].isin([1051,1052]),loyalty_df['Receipt DateTime'] + pd.Timedelta(hours=1),loyalty_df['Receipt DateTime'])
loyalty_df['time']=pd.to_datetime(loyalty_df['time'])
loyalty_df['time'] = loyalty_df['time'].dt.time


loyalty_df=loyalty_df[(loyalty_df['Menu Item Type']=='M') & (loyalty_df['StoreId'].isin([1001, 1003, 1005, 1006, 1008, 1051,1052]))]


# # Concatenate 'StoreID', 'Date', and 'CheckTime' to create 'Key'
loyalty_df['Key'] = loyalty_df['StoreId'].astype(str) + "_" + loyalty_df['date'].astype(str) + "_" + loyalty_df['time'].astype(str)



