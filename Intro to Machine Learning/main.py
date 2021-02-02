from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# Create a model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the data
iowa_model.fit(train_X, train_y)

# Predict
val_predictions = iowa_model.predict(val_X)

print(val_predictions[:5])
print(val_y.head().tolist())
print(mean_absolute_error(val_predictions, val_y))