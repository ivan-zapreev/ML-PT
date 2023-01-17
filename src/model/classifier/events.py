import numpy as np
import pandas as pd

from src.utils.logger import logger

class EventsClassifier():
    NOIZE_CLASS = -1
    
    def __init__(self, cu_tfidf, mvn_tfidf, mvss_tfidf, mvv_tfidf, scaler, pca, classifier):
        self.cu_tfidf = cu_tfidf
        self.mvn_tfidf = mvn_tfidf
        self.mvss_tfidf = mvss_tfidf
        self.mvv_tfidf = mvv_tfidf
        self.scaler = scaler
        self.pca = pca
        self.classifier = classifier
    
    def classify_events(self, events_df):
        # Default initialize the result
        y = np.full((len(events_df)), self.NOIZE_CLASS)

        # TODO: Implement
        
        # Return the resulting dataframe
        return pd.DataFrame({'EVENT_ID' : events_df['EVENT_ID'].values, 'LABEL_PRED' : y})