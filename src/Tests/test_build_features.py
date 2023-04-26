'''
This is a testing file for testing the src/features/build_features scripts.
To run this file enter into console: py ./src/tests/test_build_features.py
'''

# Import files and bibs
from src.features import build_features
import pandas as pd
import pandas.testing as pd_test
import unittest

class TestBuildFeatures(unittest.TestCase):

    def test_filter_values_by_threshold(self):
        train_df = pd.read_csv('data/raw/train_values.csv')
        print(train_df.head())
        
        # Count the number of unique values in each row of the 'geo_level_1_id' column
        #train_df['count'] = train_df['geo_level_1_id'].count()

        # Print the first 10 rows of the DataFrame with the 'count' column
        #print(train_df[['geo_level_1_id', 'count']].head(10))
        filter_values = build_features.filter_values_by_threshold(train_df, 0.98)
        
        self.assertListEqual(filter_values['count_floors_pre_eq'], [4,5,6,7,8,9])


if __name__ == '__main__':
    unittest.main()

# Starting the tests
#print("Starting tests for build_features.py")
#test_filter_values_by_threshold()