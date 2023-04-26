#################################
# Tutorial for Testing in python
# https://realpython.com/python-testing/
################################

# Import files and bibs
from src.features import build_features
import pandas as pd

def test_filter_values_by_threshold():
    train_df = pd.read_csv('data/raw/train_values.csv')

    build_features.filter_values_by_threshold(train_df, -1)
    
# Starting the tests
print("Starting tests for build_features.py")
test_filter_values_by_threshold()