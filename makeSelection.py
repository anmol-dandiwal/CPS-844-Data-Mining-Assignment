import pandas as pd

data = pd.read_csv("pulsars.csv")

# Enter in the columns that the Attribute Selection method returns.
# RecursiveFeatureElimination return [2, 3, 4, 5]
# TreeBasedSelection returns [0, 2, 3]
# 8 is included because it's the classification class.
selected_features = [2, 3, 4, 5, 8]
selected_data = data.iloc[:, selected_features]
# Save the new dataset as a CSV file
selected_data.to_csv("recursiveFeatureEliminationPulsar.csv", index=False)