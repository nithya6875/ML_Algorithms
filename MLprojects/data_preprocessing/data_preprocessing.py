import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv("Data.csv")

# dependent variables and features. Features determine dependent variable:

# matrix of features and matrix of dependent variable

X = dataset.iloc[:, : -1].values

y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])



