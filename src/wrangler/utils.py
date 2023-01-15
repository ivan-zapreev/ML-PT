import ipaddress
import Levenshtein

import regex as re
import numpy as np
import pandas as pd

from src.utils.logger import logger

# According to:
#    https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent
# The valid user agent value value shall be:
#    User-Agent: <product> / <product-version> <comment>
# Common format for web browsers:
#    User-Agent: Mozilla/5.0 (<system-information>) <platform> (<platform-details>) <extensions>
# Let us take the user agent regexp from:
#    https://regex101.com/library/2McsiK
USER_AGENT_REG_EXP = re.compile(r'\((?<info>.*?)\)(\s|$)|(?<name>.*?)\/(?<version>.*?)(\s|$)')

def is_bad_user_agent(user_agent):
    return not USER_AGENT_REG_EXP.match(user_agent)

def is_good_user_agent(user_agent):
    return bool(USER_AGENT_REG_EXP.match(user_agent))

DATA_DELIMITER = r"\';\'"

get_first_val = lambda lst: lst[0]
get_second_val = lambda lst: lst[1] if len(lst) > 1 else ''
split_data_values = lambda val: val.split(DATA_DELIMITER)

# NOTE: It would be good to have a complete list or a dynamic collection
#       For now this is fine but for production is to be more robust
LEARNED_SRC_VALUES = ['REQUEST_URI', 'REQUEST_GET_ARGS', 'REQUEST_PATH', \
                      'REQUEST_HEADERS', 'REQUEST_METHOD', 'REQUEST_COOKIES', \
                      'REQUEST_ARGS_KEYS', 'RESPONSE_HEADERS', 'REQUEST_POST_ARGS', \
                      'REQUEST_JSON', 'REQUEST_XML', 'REQUEST_ARGS', 'CLIENT_USERAGENT', \
                      'RESPONSE_BODY', 'CLIENT_SESSION_ID', 'REQUEST_CONTENT_TYPE', \
                      'REQUEST_QUERY', 'REQUEST_FILES', 'CLIENT_IP']

def get_sec_src_value_substitute(sec_src_val, dist_src_vals=LEARNED_SRC_VALUES):
    if sec_src_val != '':
        candidates = [(Levenshtein.distance(sec_src_val, src_val), src_val) for src_val in dist_src_vals if src_val.startswith(sec_src_val)]
        if candidates:
            score, substitute = min(candidates)
            logger.info(f'Found the substitute: "{sec_src_val}" -> "{substitute}", penalized Levenshtein similarity score: {score}')
        else:
            logger.warning(f'Could not find any substitute candidates for: {sec_src_val}, keeping as is!')
            substitute = sec_src_val
    else:
        # The secondaty value is not present, ignoring
        substitute = ''

    return substitute

# The list of string-valued columns
STRING_COLUMNS = ['CLIENT_IP', 'CLIENT_USERAGENT', 'MATCHED_VARIABLE_SRC', 'MATCHED_VARIABLE_NAME', 'MATCHED_VARIABLE_VALUE', 'EVENT_ID']

# Adjust column types, convert the int columns into floats for now as there are NaN values
# Note that the integer columns have valid value ranges, if outside the range the value is not properly set
INT_COLUMNS = {'REQUEST_SIZE' : {'vmin':0, 'vmax':None}, 'RESPONSE_CODE' : {'vmin':100, 'vmax':599}}

# Do the safe conversion to float/nan make sure the values are integers
# Note that, both int columns are NOT controlled by the client and thus
# if we have invalid valus in them it is safe to put them to NaN as such
# invalid values are a likely indicator of improperly collected data
def float_converter(val, vmin, vmax):
    cut_limits = lambda val, vmin, vmax : np.nan if ((vmin is not None) and (val < vmin)) or ((vmax is not None) and (val > vmax)) else val
    try:
        result = np.nan if pd.isna(val) else cut_limits(int(val), vmin, vmax)
    except ValueError:
        result = np.nan
    return result

def convert_ipv4_to_ipv6(address):
    try:
        ip_obj = ipaddress.ip_address(address)
        if isinstance(ip_obj, ipaddress.IPv4Address):
            result = ipaddress.IPv6Address('2002::' + address).exploded
        else:
            result = ip_obj.exploded
    except ValueError:
        result = address
    
    return result

def check_invalid_ip_addresses(unique_client_ips):
    invalid_ips = []
    for address in unique_client_ips:
        try:
            ipaddress.ip_address(address)
        except ValueError:
            invalid_ips.append(address)

    logger.info(f'Found {len(invalid_ips)} invalid IP addresses:\n{invalid_ips}')

def fix_missing_ipv6_zeroes(address):
    elements = address.split(':')
    elements = ['0'*(4-len(elem)) + elem if len(elem) < 4 else elem for elem in elements]
    while len(elements) < 8:
        elements.append('0000')
    return ':'.join(elements)

def _replace_variable_names_val(val, match):
    if match != '':
        val = val.replace(match + '.', '')
        val = val.replace(match, '')
    return val

def clean_up_variable_names(name, src_values):
    for src_value in src_values:
        name = _replace_variable_names_val(name, src_value)

    return name

FIX_NAME_MAPPING = {'_' : '_JQUERY_NO_CHACHE_'}

def fix_variable_names(names):
    return [FIX_NAME_MAPPING[name] if name in FIX_NAME_MAPPING else name for name in names]
        
            