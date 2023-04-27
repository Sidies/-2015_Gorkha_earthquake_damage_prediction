'''
This is a testing file for testing the src/features/build_features scripts.
To run this file enter into console: py ./src/tests/test_build_features.py

Tutorial for Testing in python
https://realpython.com/python-testing/
'''

# Import files and bibs
from src.features import build_features
import pandas as pd
import pandas.testing as pd_test
import unittest

class TestBuildFeatures(unittest.TestCase):

    def test_filter_values_by_threshold(self):
        train_df = pd.read_csv('data/raw/train_values.csv')
        filter_values = build_features.filter_values_by_threshold(train_df, 0.98)
        
        self.assertListEqual(filter_values['count_floors_pre_eq'], [4,5,6,7,8,9])
        self.assertTrue(all(value > 50 for value in filter_values['age']))
        self.assertNotIn('r', filter_values['foundation_type'])


if __name__ == '__main__':
    unittest.main()
