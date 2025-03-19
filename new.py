
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('newDATAset.csv')

df.head()

df.dropna()

df.describe()

df.dtypes

df

cod = LabelEncoder()

df["Extracurricular Activities"] = cod.fit_transform(df["Extracurricular Activities"])

#df["Extracurricular Activities Encoded"] = df["Extracurricular Activities"].apply(lambda x:1 if x == 'No' else 0)

df

X = df.drop(["Performance Index"],axis = 1)
y = df["Performance Index"]



lr = LinearRegression()

lr.fit(X,y)

lr.intercept_

lr.coef_

lr.predict(X)

