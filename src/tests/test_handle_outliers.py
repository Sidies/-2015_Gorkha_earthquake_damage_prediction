'''
This is a testing file for testing the src/features/build_features scripts.
To run this file enter into console: py ./src/tests/test_handle_outliers.py

Tutorial for Testing in python
https://realpython.com/python-testing/
'''

# Import files and bibs
import pandas as pd
import pandas.testing as pd_test
import unittest
import argparse

from src.features import handle_outliers
from sklearn.pipeline import Pipeline
from src.data import configuration as config
from src.features import build_features

class TestBuildFeatures(unittest.TestCase):

    def test_filter_values_by_threshold(self):
        train_df = pd.read_csv('data/raw/train_values.csv')
        handle_outliers.find_outliers_by_threshold(train_df, 0.02, False)

        #self.assertListEqual(filter_values['count_floors_pre_eq'], [4,5,6,7,8,9])
        #self.assertTrue(all(value > 50 for value in filter_values['age']))
        #self.assertNotIn('r', filter_values['foundation_type'])

        test1 = handle_outliers.find_outliers_by_threshold(train_df, 0.7, False)
        self.assertIn('1', str(test1['has_superstructure_mud_mortar_brick'])), f"expected value 1 in {test1['has_superstructure_mud_mortar_brick']}"
        
        
    def test_get_and_remove_outliers(self):
        train_df = pd.read_csv('data/raw/train_values.csv')
        
        outliersIndizes = handle_outliers.get_outlier_rows_as_index(train_df, ['age'], [])
        #self.assertTrue(np.isin(train_df.index[train_df['age'] == 995], outliersIndizes).all()), "expected all indizes for age 995 to be with the outlier indizes"
                
        # rows we found to contain outliers which can therefore be dropped
        # remove the has_secondary_use and has_superstructure columns to not run analysis on them
        new_categorical_columns = list(set(config.categorical_columns) - set(config.has_superstructure_columns) - set(config.has_secondary_use_columns))
        oldSize = len(train_df)
        row_indizes_to_remove = handle_outliers.get_outlier_rows_as_index(train_df, config.numerical_columns, new_categorical_columns, 0.2)
        
        train_df = build_features.remove_rows_by_integer_index(train_df, row_indizes_to_remove)
        newSize = len(train_df)
        self.assertLess(newSize, oldSize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mi', action='store_true', help='If this flag is set to true, more information about the pipeline progress will be displayed')
    args = parser.parse_args()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)