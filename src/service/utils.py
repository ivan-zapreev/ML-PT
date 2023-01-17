import json
import pandas as pd

from src.utils.logger import logger

from src.wrangler.utils import get_second_val
from src.wrangler.utils import DATA_DELIMITER
from src.wrangler.utils import split_data_values
from src.wrangler.utils import fix_variable_names
from src.wrangler.utils import is_good_user_agent
from src.wrangler.utils import adjust_column_types
from src.wrangler.utils import convert_ipv4_to_ipv6
from src.wrangler.utils import clean_up_variable_names
from src.wrangler.utils import fix_broken_ipv6_addresses
from src.wrangler.utils import get_sec_src_value_substitute

def request_data_to_df(req_data):
    # Extract the data
    data = [json.loads(elem['data']) for elem in req_data]
    
    # Convert to dataframe
    events_df = pd.DataFrame.from_records(data)
    
    logger.info(f'Resulting events dataframe columns: {events_df.columns.values}')
    
    return events_df

def wrangle_raw_data(events_df):
    # First adjust column types
    events_df = adjust_column_types(events_df)
    
    # Convert IPs to IPv6
    events_df['CLIENT_IP'] = events_df['CLIENT_IP'].apply(convert_ipv4_to_ipv6)
    
    # Fix the IPv6 addresses missing the digits and convert to int values for future ease of use
    events_df['CLIENT_IP'] = events_df['CLIENT_IP'].apply(fix_broken_ipv6_addresses)

    # Extend data with the validity flag for the user agent as it may be a good indicator of the attack
    # The percent of malformed user agent values is significantly higher for the IP from which most requests came
    events_df['IS_USERAGENT_VALID'] = events_df['CLIENT_USERAGENT'].apply(is_good_user_agent)
    
    # Fix the MATCHED_VARIABLE_NAME column values
    events_df['MATCHED_VARIABLE_SRC'] = events_df['MATCHED_VARIABLE_SRC'].apply(split_data_values)
    dist_sec_src_vals = events_df['MATCHED_VARIABLE_SRC'].apply(get_second_val).unique()
    sec_to_src_map = {sec_src_val : get_sec_src_value_substitute(sec_src_val) for sec_src_val in dist_sec_src_vals}
    fix_second_value = lambda lst : [lst[0], sec_to_src_map[lst[1]]] if len(lst) > 1 else lst
    events_df['MATCHED_VARIABLE_SRC'] = events_df['MATCHED_VARIABLE_SRC'].apply(fix_second_value)
    
    # Remove the repeated MATCHED_VARIABLE_SRC column values from MATCHED_VARIABLE_NAME
    events_df['MATCHED_VARIABLE_NAME'] = events_df.apply(clean_up_variable_names, axis=1)
    # Split into distinct variable names and fix
    events_df['MATCHED_VARIABLE_NAME'] = events_df['MATCHED_VARIABLE_NAME'].apply(lambda names : fix_variable_names(names))

    # Re-join the fields that were split
    events_df['MATCHED_VARIABLE_SRC'] = events_df['MATCHED_VARIABLE_SRC'].apply(lambda lst: DATA_DELIMITER.join(lst))
    events_df['MATCHED_VARIABLE_NAME'] = events_df['MATCHED_VARIABLE_NAME'].apply(lambda lst: DATA_DELIMITER.join(lst))
    
    # Infer the best data types
    events_df = events_df.convert_dtypes()
    
    logger.info(f'Wrangled DataFrame data types:\n{events_df.dtypes}')

    return events_df