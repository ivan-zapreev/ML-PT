import os
import bz2
import pickle

import numpy as np
import _pickle as cPickle

from src.utils.logger import logger

# Pickle a file and then compress it into a file with extension 
def compressed_pickle(folder, file_name, data):
    file_path = os.path.join(folder, f'{file_name}.pbz2')
    logger.info(f'Dumping compressed pickle file into: {file_path}')
    with bz2.BZ2File(file_path, 'w') as f: 
        cPickle.dump(data, f)
    logger.info(f'File dumping is done!')

def decompress_pickle(folder, file_name):
    file_path = os.path.join(folder, f'{file_name}.pbz2')
    logger.info(f'Loading compressed pickle file from: {file_path}')
    data = bz2.BZ2File(file_path, 'rb')
    data = cPickle.load(data)
    logger.info(f'File loading and data extraction are done!')
    return data

def store_numpy_zc(file_path, **kwds):
    logger.info(f'Dumping compressed numpy z file into: {file_path}')
    np.savez_compressed(file_path, **kwds)
    logger.info(f'File dumping is done!')

def load_numpy_zc(file_path, names):
    logger.info(f'Loading compressed numpy z file from: {file_path}')
    loaded = np.load(file_path)
    data = (loaded[name] for name in names)
    logger.info(f'File loading and data extraction are done!')
    return data

def check_file_present(file_dir, data_type, data_struct, version):
    file_path = os.path.join(file_dir, f'{data_type}_{data_struct}_v{version}.npz')
    logger.info(f'Checking if the dataset v{version} "{data_struct}" {data_type} data is pre-computed: {file_path}')
    is_not_present = not os.path.exists(file_path)
    return file_path, is_not_present