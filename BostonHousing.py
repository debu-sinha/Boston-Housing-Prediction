# Databricks notebook source
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler as sc 
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

%matplotlib inline

# COMMAND ----------

input_df = pd.read_csv("./dataset/housing.csv")

# COMMAND ----------

input_df.hist(figsize=(10, 10))

# COMMAND ----------

sns.pairplot(input_df)

# COMMAND ----------

sns.heatmap(input_df.corr())

# COMMAND ----------

# MAGIC %md
# MAGIC It seems like there is a positive correlation between median value of a house and the RM. We can also see a positive correlation between PTRATIO and LSTAT. LSTAT has a string negative correlation to MEDV.

# COMMAND ----------

input_df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC There are no null values

# COMMAND ----------

(input_df<=0.0).sum()

# COMMAND ----------

# MAGIC %md
# MAGIC There is no 0 present for values in the dataset.

# COMMAND ----------

scalar = sc()
features  = ["RM", "LSTAT"]
outcomes = ["MEDV"]

# COMMAND ----------

X = input_df[features]
y = input_df[outcomes]

# COMMAND ----------

X_scaled = scalar.fit_transform(X)

# COMMAND ----------

X = pd.DataFrame(data = X_scaled, columns=np.array(features))

# COMMAND ----------

kfold = 10
kf = KFold(n_splits=10, random_state= 10)

model_names = ["Linear Regression","Ridge Regression", "Elastic Net"]
models = [linear_model.LinearRegression(), linear_model.Ridge(alpha=.5, random_state=10), linear_model.ElasticNet(random_state=10)]

mean_r2 = []
global_r2 = []

def evaluate_models(X=X, y=y):
   '''
       This method performs 10 fold cross validation on list of models and returns a dataframe with mean f1 and mean accuracy scores for each model
   '''
   for name, model in zip(model_names, models):
        r2 = []
        
        for train_index, test_index in kf.split(X, y):
            
            X_train = X.loc[train_index] 
            y_train = y.loc[train_index]
            X_test = X.loc[test_index]
            y_test = y.loc[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2.append(r2_score(y_test, y_pred))
            
        global_r2.append(r2)
        mean_r2.append(np.mean(np.array(r2)))
        
   model_perf_df = pd.DataFrame(np.array([ mean_r2]).T,index=model_names)   
   model_perf_df.columns = ['Mean R2']
   return model_perf_df

# COMMAND ----------

def generate_box_plot():
    box=pd.DataFrame( data=global_r2,index=[model_names])
    plt.figure(figsize=(20, 20))
    sns.boxplot(data=box.T).set_title('Mean r2 score')

# COMMAND ----------

## define utility methods
def convert_to_dataframe(ndar, cols):
    '''Given a set of records in nupy array and a list of column names return a Pandas dataframe'''
    pdf = pd.DataFrame.from_records(ndar)
    pdf.columns = cols
    return pdf

# COMMAND ----------

evaluate_models().sort_values(ascending=False, by = 'Mean R2')

# COMMAND ----------

global_r2

# COMMAND ----------

generate_box_plot()

# COMMAND ----------


