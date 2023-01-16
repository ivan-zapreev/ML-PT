import os
import bz2
import pickle

import numpy as np
import _pickle as cPickle

from src.utils.logger import logger

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
