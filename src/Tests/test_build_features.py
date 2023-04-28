'''
This is a testing file for testing the src/features/build_features scripts.
To run this file enter into console: py ./src/tests/test_build_features.py

Tutorial for Testing in python
https://realpython.com/python-testing/
'''

# Import files and bibs
import pandas as pd
import pandas.testing as pd_test
import unittest
import argparse

from src.features import build_features
from sklearn.pipeline import Pipeline
from src.features.build_features import DropRowsTransformer

class TestBuildFeatures(unittest.TestCase):

    def test_filter_values_by_threshold(self):
        train_df = pd.read_csv('data/raw/train_values.csv')
        build_features.find_outliers_by_threshold(train_df, 0.02, args.mi)
        
        #self.assertListEqual(filter_values['count_floors_pre_eq'], [4,5,6,7,8,9])
        #self.assertTrue(all(value > 50 for value in filter_values['age']))
        #self.assertNotIn('r', filter_values['foundation_type'])
        
        test1 = build_features.find_outliers_by_threshold(train_df, 0.7)
        self.assertIn('1', str(test1['has_superstructure_mud_mortar_brick'])), f"expected value 1 in {test1['has_superstructure_mud_mortar_brick']}"

    def test_DropRowsTransformer(self):
        # create a sample dataframe
        df = pd.DataFrame({
            'A': ['a', 'b', 'b', 'c', 'd', 'e'],
            'B': [1, 2, 3, 4, 4, 5],
            'C': [6, 7, 8, 9, 9, 10]
        })

        # create a dictionary of rows to drop
        rows_to_drop = {'A': ['a', 'b', 'e'], 'B': [1, 4, 5], 'C': [6, 9, 10]}

        # create a pipeline with the DropRowsTransformer
        pipeline = Pipeline([
            ('drop_rows', DropRowsTransformer(rows_to_drop))
        ])

        # apply the pipeline to the dataframe
        new_df = pipeline.fit_transform(df)

        # create a test dataframe with the expected values
        test_df = pd.DataFrame({
            'A': ['b', 'c'],
            'B': [2, 3],
            'C': [7, 8]
        })

        # check if the new dataframe matches the test dataframe
        assert new_df.equals(test_df),  f"Expected: \n {test_df} \n but got: \n {new_df}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mi', action='store_true', help='If this flag is set to true, more information about the pipeline progress will be displayed')
    args = parser.parse_args()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)