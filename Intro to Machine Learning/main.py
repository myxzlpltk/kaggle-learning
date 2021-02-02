import pandas as pd

# Path of the file to read
iowa_file_path = 'data/home-data-for-ml-course/train.csv'

# Read data
home_data = pd.read_csv(iowa_file_path)

# Avg Lot Size
print(round(home_data['LotArea'].mean()))

# least home age
print(pd.to_datetime('now').year - home_data['YearBuilt'].max())