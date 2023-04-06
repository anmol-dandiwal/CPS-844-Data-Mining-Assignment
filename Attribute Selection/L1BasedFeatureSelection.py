import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

# Load the dataset
url = "pulsars.csv"
df = pd.read_csv(url, header=None)

# Split the dataset into input features and target variable
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Define the logistic regression model with L1 penalty
clf = LogisticRegression(penalty='l1', solver='liblinear')

# Perform a grid search to find the optimal C value
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(clf, param_grid=params, cv=5)
grid_search.fit(x, y)
best_c = grid_search.best_params_['C']

# Use SelectFromModel to perform feature selection with the optimal C value
sfm = SelectFromModel(LogisticRegression(penalty='l1', C=best_c, solver='liblinear'))
X_new = sfm.fit_transform(x, y)

# Print the selected features
selected_features = x.columns[sfm.get_support()]
print("Selected features:", selected_features)
