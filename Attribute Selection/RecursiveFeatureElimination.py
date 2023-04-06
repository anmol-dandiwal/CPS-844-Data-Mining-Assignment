from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Load the dataset
url = "pulsars.csv"
df = pd.read_csv(url, header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LogisticRegression()

#Create the RFE model that will select 4 features that are the most important.
'''
    Args:
    model:
    n_features_to_select:
    step:
'''
selector = RFE(model, n_features_to_select=4, step=5)

# Fit the selector to the data
selector.fit(X, y)
selected_features = np.array(df.columns[:-1])[selector.support_]

# Print the selected features
print("Selected features:", selected_features)