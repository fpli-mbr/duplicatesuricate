import pandas as pd


class Hardcodedlinkage:
    def __init__(self,traincols):
        self.traincols = traincols
        pass
    def fit(self):
        pass
    def predict_proba(self,x_score):
        return (x_score).fillna(0).mean(axis = 1)
