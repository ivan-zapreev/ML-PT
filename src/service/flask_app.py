import os
import sys
import argparse

import pandas as pd

from flask import Flask
from flask import request
from flask import jsonify

# Instantiate Flask app
FLASK_APP = Flask(__name__)

# Make sure we have the proper package folder in the path
server_file_folder = os.path.dirname(os.path.abspath(__file__))
print(f'The server source file folder is: {server_file_folder}')
package_root = os.path.join(server_file_folder, '..', '..')
print(f'Adding package root folder to path: {package_root}')
sys.path.append(package_root)

from src.utils.logger import logger
from src.service.utils import wrangle_raw_data
from src.service.utils import request_data_to_df
from src.utils.file_utils import load_pickle_data

# Define the default server name
_SERVER_HOST_DEF = 'localhost'
# Define the default value for the server port
_SERVER_PORT_DEF = 8080

# The default data folder to get the data from
_DATA_FOLDER_DEF = os.path.join('.', 'data')

# Stores the default names for the input files needed
_SERVER_PKL_FILES = {'classifier': True, 'extractor': True}

def __obtain_arguments_parser():
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-sh", "--server_host", required=False, default = _SERVER_HOST_DEF,
                    help="The server host to bind to, defaults to: {_SERVER_HOST_DEF}")
    
    ap.add_argument("-sp", "--server_port", required=False, default = _SERVER_PORT_DEF,
                    help="The TCP port the server will listen to, defaults to: {_SERVER_PORT_DEF}")
    
    ap.add_argument("-df", "--data_folder", required=False, default = _DATA_FOLDER_DEF,
                    help=f"The folder to read pre-trianed (.pkl) sub-models from, defaults to {_DATA_FOLDER_DEF}")
    
    return vars(ap.parse_args())

def __obtain_script_args():
    args = __obtain_arguments_parser()

    server_host = args['server_host']
    logger.info(f'Starting TCP Classifier server on: {server_host}')

    server_port = args['server_port']
    logger.info(f'Starting TCP Classifier server on port: {server_port}')

    data_folder = args['data_folder']
    logger.info(f'The pre-trined sub-models data folder: {data_folder}')

    return server_host, server_port, data_folder

def __load_pre_trained_models(data_folder):
    logger.info(f'Start loading the Classifier pre-trained sub-models from: {data_folder}')
    models = {}
    for file_name, is_compulsory in _SERVER_PKL_FILES.items():
        data, file_path = load_pickle_data(data_folder, file_name)
        if data is None and is_compulsory:
            raise Exception(f'Missing compulsory server input file: {file_path}')
        else:
            models[file_name] = data

    return models

def classify_events(extractor, classifier, events_df):
    # Extract features
    X = extractor.transform(events_df)
    logger.info(f'The computed feature space shape is: {X.shape}')    

    # Classify events
    y = classifier.predict(X)

    # Return the resulting dataframe
    return pd.DataFrame({'EVENT_ID' : events_df['EVENT_ID'].values, 'LABEL_PRED' : y})

@FLASK_APP.post("/predict")
def handle_prediction_requests():
    res_data, res_code = None, None
    if request.is_json:
        req_data = request.get_json()
        logger.debug(f'Handling new JSON request!')
        
        try:
            # Parse the request data
            events_df = request_data_to_df(req_data)
            
            logger.info(f'Obtained request events data frame:\n{events_df.head(10)}')
            
            # Wrangle data
            events_df = wrangle_raw_data(events_df)

            # Classify the events
            classes_df = classify_events(FLASK_APP.config['extractor'], FLASK_APP.config['classifier'], events_df)            

            # Set the result
            res_data, res_code = jsonify(classes_df.to_dict('records')), 200
        except:
            message = 'Failed classifying the provided events!'
            logger.exception(message)
            res_data, res_code = {'error': message}, 500
    else:
        res_data, res_code = {'error': 'Request must be JSON'}, 415
    
    return res_data, res_code

if __name__ == '__main__':
    # Get the script arguments
    server_host, server_port, data_folder = __obtain_script_args()

    # Load the necessary files from
    models = __load_pre_trained_models(data_folder)
    
    # Run the Flask Server
    logger.info(f'Starting the Flask server')
    FLASK_APP.config['extractor'] = models['extractor']
    FLASK_APP.config['classifier'] = models['classifier']
    FLASK_APP.run(host=server_host, port=server_port)

