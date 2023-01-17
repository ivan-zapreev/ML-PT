import os
import pickle

import numpy as np

from src.utils.logger import logger

def dump_pickle_data(folder_path, file_name, data):
    file_path = os.path.join(folder_path, f'{file_name}.pkl')
    logger.info(f'Dumping pickle file into: {file_path}')
    with open(file_path, 'wb') as pfile:
        pickle.dump(data, pfile, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_data(folder_path, file_name):
    file_path = os.path.join(folder_path, f'{file_name}.pkl')
    logger.info(f'Loading pickle data from: {file_path}')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as pfile:
            data = pickle.load(pfile)
    else:
        logger.warning(f'The required file: {file_path} is missing!')
        data = None
    return data, file_path

def store_numpy_zc(file_path, **kwds):
    logger.info(f'Dumping compressed numpy z file into: {file_path}.npz')
    np.savez_compressed(file_path, **kwds)
    logger.info(f'File dumping is done!')

def load_numpy_zc(file_path, names):
    logger.info(f'Loading compressed numpy z file from: {file_path}')
    loaded = np.load(file_path)
    data = (loaded[name] for name in names)
    logger.info(f'File loading and data extraction are done!')
    return data
