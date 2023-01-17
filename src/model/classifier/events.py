from src.utils.logger import logger

class EventsClassifier():
    
    def __init__(self, cu_tfidf, mvn_tfidf, mvss_tfidf, mvv_tfidf, scaler, pca, classifier):
        self.cu_tfidf = cu_tfidf
        self.mvn_tfidf = mvn_tfidf
        self.mvss_tfidf = mvss_tfidf
        self.mvv_tfidf = mvv_tfidf
        self.scaler = scaler
        self.pca = pca
        self.classifier = classifier
    
