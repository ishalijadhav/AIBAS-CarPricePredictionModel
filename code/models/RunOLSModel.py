import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle


model_path = "currentOlsSolution.pkl"
#model_path = "/tmp/knowledgeBase/currentOlsSolution.pkl"
with open(model_path, 'rb') as model_file:
    ols_model = pickle.load(model_file)

activation_data = pd.read_csv("../../data/activation_data.csv")
#activation_data = pd.read_csv("/tmp/activationBase/activation_data.csv")

activation_data = activation_data.drop(columns=activation_data.columns[6])
activation_data = sm.add_constant(activation_data,has_constant='add')
#print(activation_data.head())

#print("activation_data.shape: ", activation_data.shape)

predictions = ols_model.predict(activation_data)
denormalized_price = predictions * (10000000 - 29999) + 29999
print(denormalized_price)