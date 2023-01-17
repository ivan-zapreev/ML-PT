import os
import sys
import argparse

# Make sure we have the proper package folder in the path
server_file_folder = os.path.dirname(os.path.abspath(__file__))
print(f'The server source file folder is: {server_file_folder}')
package_root = os.path.join(server_file_folder, '..', '..')
print(f'Adding package root folder to path: {package_root}')
sys.path.append(package_root)

from src.utils.logger import logger
from src.utils.file_utils import load_pickle_data
from src.service.socker_server import SocketServer
from src.model.classifier.events import EventsClassifier

# Define the default server name
_SERVER_NAME_DEF = 'localhost'

# Define the default value for the server port
_SERVER_PORT_DEF = 8080

# The default data folder to get the data from
_DATA_FOLDER_DEF = os.path.join('.', 'data')

# Stores the default names for the input files needed
_SERVER_PKL_FILES = {'classifier': True, 'cu_tfidf': True, 'mvn_tfidf': True, 'mvss_tfidf': True, 'mvv_tfidf': True, 'scaler': False, 'pca': False}

def __obtain_arguments_parser():
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-sn", "--server_name", required=False, default = _SERVER_NAME_DEF,
                    help="The server name to bind to, defaults to: {_SERVER_NAME_DEF}")
    
    ap.add_argument("-sp", "--server_port", required=False, default = _SERVER_PORT_DEF,
                    help="The TCP port the server will listen to, defaults to: {_SERVER_PORT_DEF}")
    
    ap.add_argument("-df", "--data_folder", required=False, default = _DATA_FOLDER_DEF,
                    help=f"The folder to read pre-trianed (.pkl) sub-models from, defaults to {_DATA_FOLDER_DEF}")
    
    return vars(ap.parse_args())

def __obtain_script_args():
    args = __obtain_arguments_parser()

    server_name = args['server_name']
    logger.info(f'Starting TCP Classifier server on: {server_name}')

    server_port = args['server_port']
    logger.info(f'Starting TCP Classifier server on port: {server_port}')

    data_folder = args['data_folder']
    logger.info(f'The pre-trined sub-models data folder: {data_folder}')

    return server_name, server_port, data_folder

def __load_classifier_pkl_files(data_folder):
    logger.info(f'Start loading the Classifier pre-trained sub-models from: {data_folder}')
    classifier_data = {}
    for file_name, is_compulsory in _SERVER_PKL_FILES.items():
        data, file_path = load_pickle_data(data_folder, file_name)
        if data is None and is_compulsory:
            raise Exception('Missing compulsory server input file: {file_path}')
        else:
            classifier_data[file_name] = data

    return classifier_data

def __run_classifier_request_handling(server_name, server_port, classifier):
    # Instantiate the server
    server = SocketServer(server_name, server_port, classifier)
    
    # Start the server
    server.start()

    # Allow to shut down on key press
    logger.info('----------------------------------------------------------------')
    logger.info('Press <ENTER> to terminate the image serving HTTP server ...')
    _ = input()
    
    # Run the shut-down sequence
    server.stop()

if __name__ == '__main__':
    # Get the script arguments
    server_name, server_port, data_folder = __obtain_script_args()

    # Load the necessary files from
    classifier_data = __load_classifier_pkl_files(data_folder)
    
    # Instantiate the classifier
    classifier = EventsClassifier(**classifier_data)
    
    # Run the TCP server
    __run_classifier_request_handling(server_name, server_port, classifier)