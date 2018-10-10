from DecisionTree import DecisionTree

from pprint import pprint
import pandas as pd
import numpy as np
import random

class BaggingTrees(object):
    
    def __init__(self, n_estimators=10, bootstrap_value=1.0, **tree_params):
        ''' Constructor. 
                n_estimators: number of bagged trees
                boostrap_value: sample size, either in percentages or number of samples
                tree_params: decision tree hyperparams
        '''
        self._tree_params = tree_params
        self._n_estimators = n_estimators
        self._estimators = [DecisionTree(**tree_params) for _ in range(n_estimators)]
        self._bootstrap_value = bootstrap_value
        
    def _bootstrap(self, df):
        ''' helper: Sampling without replacement '''            
        dados = df.values
        samples = []
        sample_size = self._bootstrap_value if self._bootstrap_value > 1 else int(self._bootstrap_value*len(dados))
        for i in range(self._n_estimators):  # One sample per estimator
            s = random.choices(dados, k = sample_size)
            s = pd.DataFrame(s, columns=df.columns.values)
            samples.append(s)
                    
        return samples
    
    def fit(self, df):
        ''' Train bagged trees '''
        samples = self._bootstrap(df)
        for i, tree in enumerate(self._estimators):
            tree.fit(samples[i])
            
    def _format_prediction(self, predictions):
        ''' helper: Formats the predictions in a easy-to-read format '''
        prediction = sum(predictions)/self._n_estimators # Average of predictions
        prediction['pred'] = prediction.idxmax(axis=1)
        cols = prediction.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        return prediction[cols]
    
    def predict(self, df):
        ''' Make predictions and return the probabilities along the class with highest probability '''
        predictions = []
        for tree in self._estimators:
            pred = tree.predict(df)
            pred = pred['proba'].apply(pd.Series).fillna(0)
            predictions.append(pred)
        
        return self._format_prediction(predictions)