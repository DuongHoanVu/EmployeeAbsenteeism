import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
import pickle

class CustomScaler(BaseEstimator,TransformerMixin): 
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X, y=None, copy=None):
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = self.scaler.transform(X[self.columns])
        X_scaled = pd.DataFrame(X_scaled, columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        return pd.concat([X_not_scaled, X_scaled], axis=1)

class absenteeism_model():
    def __init__(self, model_file, scaler_file):
        with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            pass
    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file,delimiter=',')

        df = df.drop(['ID'], axis = 1)

        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
        df = df.drop(['Reason for Absence'], axis = 1)
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)

        df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')
        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)
        df['Month Value'] = list_months

        def date_to_weekday(date_value):
            return date_value.weekday()
        df['Day of the Week'] = df['Date'].apply(date_to_weekday)

        df = df.drop(['Date'], axis = 1)

        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

        df = df.fillna(value=0)

        self.preprocessed_data = df.copy()
        self.data = self.scaler.transform(df)

    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):  
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred

    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    # predict the outputs and the probabilities and 
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data
            