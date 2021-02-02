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

# best_max_leaf_node
best_max_leaf_node = 70

# Create a final model
final_model = DecisionTreeRegressor(max_leaf_nodes=best_max_leaf_node)

# Fit model
final_model.fit(train_X, train_y)

# Predict
predictions = final_model.predict(val_X)
mae = mean_absolute_error(predictions, val_y)
print(mae)

# Create Random Forest Model
rf_model = RandomForestRegressor()

# Fitting model
rf_model.fit(train_X, train_y)

# Get absolute error
rf_val_mae = mean_absolute_error(rf_model.predict(val_X), val_y)
print(rf_val_mae)

# Train all data
rf_model_on_full_data = RandomForestRegressor(random_state=1)
# Fit data
rf_model_on_full_data.fit(X, y)

# Test data path
test_data_path = './data/home-data-for-ml-course/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': test_preds
})