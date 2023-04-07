from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load the dataset
url = "../pulsars.csv"
df = pd.read_csv(url, header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LogisticRegression()

#Create the RFE model that will select 4 features that are the most important.
selector = RFE(model, n_features_to_select=4, step=1)

# Fit the selector to the data
selector.fit(X, y)
selectedFeatures = np.array(df.columns[:-1])[selector.support_]

# Print the selected features
print("Selected features:", selectedFeatures)