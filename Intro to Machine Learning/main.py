from sklearn.metrics import mean_absolute_error
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

# Create model based on features
iowa_model = DecisionTreeRegressor(random_state=5)

# Fit the model
# X = Prediction features
# y = prediction result [SalePrice]
iowa_model.fit(X, y)

# Predict X itself
predictions = iowa_model.predict(X)

# Get absolute errors between datasets and predictions
# error = actual − predicted
error = mean_absolute_error(y, predictions)