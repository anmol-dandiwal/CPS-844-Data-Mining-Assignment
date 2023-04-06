import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# Load the dataset
url = "pulsars.csv"
df = pd.read_csv(url, header=None)

# Split the dataset into input features and target variable
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Define the tree-based feature selection model
clf = ExtraTreesClassifier(n_estimators=100)

# Fit the model and select features
clf.fit(X, y)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)

# Print the selected features
selected_features = X.columns[model.get_support()]
print("Selected features:", selected_features)