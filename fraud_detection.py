import pandas as pd
from pyod.models.auto_encoder import AutoEncoder

# Load dataset
data = pd.read_csv("creditcard.csv")

# Remove target column if present
X = data.drop("Class", axis=1)

# Create AutoEncoder model
from pyod.models.auto_encoder import AutoEncoder

clf = AutoEncoder(
    hidden_neuron_list=[64, 32, 32, 64],
    epoch_num=10,
    batch_size=32,
    contamination=0.001
)

clf.fit(X)

predictions = clf.labels_

print("Total Transactions:", len(X))
print("Fraudulent Transactions Detected:", sum(predictions))