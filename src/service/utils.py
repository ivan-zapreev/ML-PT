import json
import pandas as pd

from src.utils.logger import logger

def request_data_to_df(req_data):
    # Extract the data
    data = [json.loads(elem['data']) for elem in req_data]
    
    # Convert to dataframe
    events_df = pd.DataFrame.from_records(data)
    
    logger.info(f'Resulting events dataframe columns: {events_df.columns.values}')
    
    return events_df

def wrangle_raw_data(events_df):
    # TODO: Implement
    return events_df