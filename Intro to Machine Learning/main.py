from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd

# Path of the file to read
iowa_file_path = 'data/home-data-for-ml-course/train.csv'

# Read data
home_data = pd.read_csv(iowa_file_path)

# Get Sale Price
y = home_data.SalePrice

# Get predictive features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Create a new Data Frame based on features
X = home_data[features]

# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    # Create a model
    model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=max_leaf_nodes)
    # Fit the data
    model.fit(train_X, train_y)
    # Predict
    predictions = model.predict(val_X)
    # Get MAE
    mae = mean_absolute_error(predictions, val_y)
    # Return mae
    return (mae)

# Get candidate
candidate_max_leaf_nodes = range(5, 505, 5)
# Testing candidate
maes = []
for max_leaf_node in candidate_max_leaf_nodes:
    maes.append((max_leaf_node, get_mae(max_leaf_node, train_X, val_X, train_y, val_y)))

best_max_leaf_node = min(maes, key = lambda sub: sub[1])[0]
# best_max_leaf_node = 70

# Plotting
plt.plot(list(map(lambda x : x[0], maes)), list(map(lambda x : x[1], maes)))
plt.title('Decision Tree Regression')
plt.xlabel('Max Leaf Nodes')
plt.ylabel('Mean Absolute Error')
plt.show()

# Create a final model
final_model = DecisionTreeRegressor(max_leaf_nodes=best_max_leaf_node)

# Fit model
final_model.fit(train_X, train_y)

# Predict
predictions = final_model.predict(val_X)
mae = mean_absolute_error(predictions, val_y)