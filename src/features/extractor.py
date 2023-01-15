import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import logger

class FeatureExtractor():
   
    # Define various column data fitters
    __cu_feature_fitter = lambda self, col_data: self.cu_tfidf.fit(col_data)
    __mvs_feature_fitter = lambda self, col_data: self.mvss_tfidf.fit(col_data)
    __mvn_feature_fitter = lambda self, col_data: self.mvn_tfidf.fit(col_data)
    __mvv_feature_fitter = lambda self, col_data: self.mvv_tfidf.fit(col_data)

    # Define Fitter mappings
    FEATURE_FIT_MAPPING = {
                              'CLIENT_USERAGENT' : __cu_feature_fitter,
                              'MATCHED_VARIABLE_SRC' : __mvs_feature_fitter,
                              'MATCHED_VARIABLE_NAME' : __mvn_feature_fitter,
                              'MATCHED_VARIABLE_VALUE' : __mvv_feature_fitter
                           }
    
    # Define various column data transformers
    __ip_feature_transformer = lambda self, col_data: np.stack(col_data.apply(lambda val: np.array([int(entry, 16) for entry in val.split(':')])).values)
    __rs_feature_transformer = lambda self, col_data: np.reshape(col_data.values, (-1, 1))
    __rc_feature_transformer = lambda self, col_data: np.reshape(col_data.values, (-1, 1))
    __cu_feature_transformer = lambda self, col_data: self.cu_tfidf.transform(col_data).toarray()
    __iuv_feature_transformer = lambda self, col_data: np.reshape(col_data.astype('int').values, (-1, 1))
    __mvs_feature_transformer = lambda self, col_data: self.mvss_tfidf.transform(col_data).toarray()
    __mvn_feature_transformer = lambda self, col_data: self.mvn_tfidf.transform(col_data).toarray()
    __mvv_feature_transformer = lambda self, col_data: self.mvv_tfidf.transform(col_data).toarray()
    
    # Define Transformer mappings
    FEATURE_TRANS_MAPPING = {
                                'CLIENT_IP' : __ip_feature_transformer,
                                'REQUEST_SIZE' : __rs_feature_transformer,
                                'RESPONSE_CODE' : __rc_feature_transformer,
                                'CLIENT_USERAGENT' : __cu_feature_transformer,
                                'IS_USERAGENT_VALID' : __iuv_feature_transformer,
                                'MATCHED_VARIABLE_SRC' : __mvs_feature_transformer,
                                'MATCHED_VARIABLE_NAME' : __mvn_feature_transformer,
                                'MATCHED_VARIABLE_VALUE' : __mvv_feature_transformer
                            }

    # Store the PCA default parameter values
    PCA_ARGS_DEFAULTS = {'n_components' : 0.999999}
    
    # Store the default ignore columns
    IGNORE_COLS_DEFAULTS = ['EVENT_ID']

    def __init__(self, ignore_columns=IGNORE_COLS_DEFAULTS, is_scale=True, pca_args=PCA_ARGS_DEFAULTS,
                 cu_max_features = 5, # Here we simply took some number to accomodate at least part of the user agent string
                 mvss_max_features = 5, # There is always at most two SRC values in the data
                 mvn_max_features=15, # Within the data at most 11 variable names were found
                 mvv_max_features=40): # Within the data at most 36 values were found
        # Remember the feature columns
        self.ignore_columns = ignore_columns
        logger.info(f'Actual Non-Feature Columns: {self.ignore_columns}')
        
        # Remember the normalization flag
        self.is_scale = is_scale
        logger.info(f'Actual Scaling flag: {self.is_scale}')
        
        # Put the argument values on top of the defaults
        if pca_args is None:
            self.pca_args = None
        else:
            self.pca_args = self.PCA_ARGS_DEFAULTS.copy()
            self.pca_args.update(pca_args)
        logger.info(f'Actual PCA arguments: {self.pca_args}, the PCA is: {"Enabled" if self.pca_args is not None else "Disabled"}')
    
        # Initialize internal class variables
        self.input_features = []
        self.cu_tfidf = TfidfVectorizer(max_features=cu_max_features)
        self.mvss_tfidf = TfidfVectorizer(max_features=mvss_max_features)
        self.mvn_tfidf = TfidfVectorizer(max_features=mvn_max_features)
        self.mvv_tfidf = TfidfVectorizer(max_features=mvv_max_features)
        self.scaler = StandardScaler() if self.is_scale else None
        self.pca = PCA(**self.pca_args) if self.pca_args is not None else None
    
    def _register_new_features(self, name, data):
        number = data.shape[1]
        for idx in range(number):
            feature_name = f'{name}_{idx}' if number > 1 else name
            if feature_name not in self.input_features:
                self.input_features.append(feature_name)
    
    # Define the feature extraction common wrapper method
    def _extract_column_features(self, data_df, data_extractor, col_name, X, is_log):
        col_data = data_extractor(self, data_df[col_name])
        X = np.append(X, col_data, axis=1)
        if is_log: logger.debug(f'Added features from: "{col_name}", the resulting X shape is: {X.shape}')

        # Register feature names
        self._register_new_features(col_name, col_data)
        
        return X
    
    def __get_feature_columns(self, data_df, is_log=False):
        feature_cols = [col_name for col_name in data_df.columns.values if col_name not in self.ignore_columns]
        
        if is_log: logger.info(f'Considering feature columns: {feature_cols}')
        
        return feature_cols
    
    def fit(self, data_df):
        logger.info(f'Start fitting the Feature Extraction model')
        
        # Fit vectorizers for the required columns
        for col_name in self.__get_feature_columns(data_df, is_log=True):
            if col_name in self.FEATURE_FIT_MAPPING:
                logger.info(f'Fitting the vectorizer for: "{col_name}"')
                col_fit = self.FEATURE_FIT_MAPPING[col_name]
                col_fit(self, data_df[col_name])
        
        # Further check if we need to scale and or do PCA
        if self.scaler is not None or self.pca is not None:
            # Get the transformed data
            X = self.__transform(data_df, is_log=True)
            
            # Fit the scaler if needed
            if self.scaler is not None:
                logger.info(f'Start fitting the scaler')
                self.scaler.fit(X)

            # Fit the PCA if needed
            if self.pca is not None:
                logger.info(f'Start fitting the PCA')
                self.pca.fit(pd.DataFrame(X, columns = self.input_features, dtype = float))
        
        logger.info(f'Fitting the Feature Extractor model is done!')

    def get_feature_names_out(self):
        if self.pca:
            pca_feature_names = self.pca.get_feature_names_out(self.input_features)
            logger.info(f'The PCA feature name out:\n{pca_feature_names}')
            
            # Prepare the components relations with features
            raw_relation_df = pd.DataFrame(self.pca.components_, columns=self.input_features, index = pca_feature_names)
            # Take the absolute values as the sign does not matter
            raw_relation_df = raw_relation_df.abs()
            if len(raw_relation_df) <= 10:
                logger.info(f'The complete PCA component feature contributions are:\n{raw_relation_df}')
            
            # Get the most relation to the PCA features
            relation_df = raw_relation_df.idxmax(axis=1)
            logger.info(f'The PCA components (absolute, maximum) relations with features:\n{relation_df}')
            
            feature_map = {f'pca{idx}' : relation_df.loc[f'pca{idx}'] for idx in range(len(pca_feature_names))}
            main_features = [feature_map[pca_name] for pca_name in pca_feature_names]
        else:
            main_features = self.input_features

        logger.info(f'The main features contributed to PCA components:\n{main_features}')
        logger.info(f'The variance explained per PCA component:\n{self.pca.explained_variance_ratio_}')
        
        return main_features, self.pca.explained_variance_ratio_, raw_relation_df
    
    def __transform(self, data_df, is_log=False):
        # Initialize the two dimensional numpy array to be used
        X = np.empty(shape=(len(data_df),0))

        # Extract the features for the specified columns
        for col_name in self.__get_feature_columns(data_df):
            col_feat_ext = self.FEATURE_TRANS_MAPPING[col_name]
            X = self._extract_column_features(data_df, col_feat_ext, col_name, X, is_log=is_log)
        
        return X

    
    def transform(self, data_df):
        logger.info(f'Start transforming the data with the Feature Extraction model')
        
        # Get the features
        X = self.__transform(data_df)
        
        # Do the Normalization if required
        if self.scaler is not None:
            logger.info(f'Starting the scaler transform')
            X = self.scaler.transform(X)
        
        # Do the PCA if required
        if self.pca is not None:
            logger.info(f'Starting the PCA transform, the initial X shape: {X.shape}')
            X = self.pca.transform(pd.DataFrame(X, columns = self.input_features, dtype = float))
            logger.info(f'The X shape after the PCA transform: {X.shape}')
       
        logger.info(f'Transforming with the Feature Extractor model is done!')
        
        return X
    
    def fit_transform(self, data_df):
        # Fit the model
        self.fit(data_df)
        
        # TODO: It is not optimal as both fit and transform will
        #       use __transform in case of Scaling or PCA enabled.
        #       One could optimize the logic to do __transform only
        #       once for better performance. This is however not
        #       critical for the amount of data at hand.
        
        # Transform the data
        return self.transform(data_df)