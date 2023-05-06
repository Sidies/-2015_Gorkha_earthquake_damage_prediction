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
from src.features.build_features import *
from src.data import configuration as config

class TestBuildFeatures(unittest.TestCase):

    def test_create_one_hot_encoding(self):
        df = pd.DataFrame({
            'a': [1, 0, 1],
            'b': [0, 1, 1]
        })

        new_df = build_features.create_one_hot_encoding(df)

        # create a test dataframe with the expected values
        test_df = pd.DataFrame({
            'a': [1, 0, 0],
            'b': [0, 1, 0],
            'a+b': [0, 0, 1]
        })

        # check if the new dataframe matches the test dataframe
        assert new_df.equals(test_df),  f"Expected: \n {test_df} \n but got: \n {new_df}"

    def test_filter_values_by_threshold(self):
        train_df = pd.read_csv('data/raw/train_values.csv')
        build_features.find_outliers_by_threshold(train_df, 0.02, False)

        #self.assertListEqual(filter_values['count_floors_pre_eq'], [4,5,6,7,8,9])
        #self.assertTrue(all(value > 50 for value in filter_values['age']))
        #self.assertNotIn('r', filter_values['foundation_type'])

        test1 = build_features.find_outliers_by_threshold(train_df, 0.7, False)
        self.assertIn('1', str(test1['has_superstructure_mud_mortar_brick'])), f"expected value 1 in {test1['has_superstructure_mud_mortar_brick']}"
        
    def test_get_outlier_rows_as_index(self):
        train_df = pd.read_csv('data/raw/train_values.csv')
        
        outliersIndizes = build_features.get_outlier_rows_as_index(train_df, ['age'], [])
        #self.assertTrue(np.isin(train_df.index[train_df['age'] == 995], outliersIndizes).all()), "expected all indizes for age 995 to be with the outlier indizes"
                
        # rows we found to contain outliers which can therefore be dropped
        # remove the has_secondary_use and has_superstructure columns to not run analysis on them
        new_categorical_columns = list(set(config.categorical_columns) - set(config.has_superstructure_columns) - set(config.has_secondary_use_columns))
        oldSize = len(train_df)
        row_indizes_to_remove = build_features.get_outlier_rows_as_index(train_df, config.numerical_columns, new_categorical_columns, 0.2)
        train_df = build_features.remove_rows_by_integer_index(train_df, row_indizes_to_remove)
        newSize = len(train_df)
        self.assertLess(newSize, oldSize)

    # def test_DropRowsTransformer(self):
    #     # create a sample dataframe
    #     df = pd.DataFrame({
    #         'A': ['a', 'b', 'b', 'c', 'd', 'e'],
    #         'B': [1, 2, 3, 4, 4, 5],
    #         'C': [6, 7, 8, 9, 9, 10]
    #     })
    #
    #     # create a dictionary of rows to drop
    #     rows_to_drop = {'A': ['a', 'b', 'e'], 'B': [1, 4, 5], 'C': [6, 9, 10]}
    #
    #     # create a pipeline with the DropRowsTransformer
    #     pipeline = Pipeline([
    #         ('drop_rows', DropRowsTransformer(rows_to_drop))
    #     ])
    #
    #     # apply the pipeline to the dataframe
    #     new_df = pipeline.fit_transform(df)
    #
    #     # create a test dataframe with the expected values
    #     test_df = pd.DataFrame({
    #         'A': ['b', 'c'],
    #         'B': [2, 3],
    #         'C': [7, 8]
    #     })
    #
    #     # check if the new dataframe matches the test dataframe
    #     assert new_df.equals(test_df),  f"Expected: \n {test_df} \n but got: \n {new_df}"

    def test_one_hot_decoder_transformer(self):
        df = pd.DataFrame({
            'a': [1, 0, 0, 0, 1],
            'b': [0, 1, 1, 0, 1],
            'c': [1, 2, 3, 4, 5]
        })
        df[['a', 'b']] = df[['a', 'b']].astype('category')
        df[['c']] = df[['c']].astype('float64')

        test_df = pd.DataFrame({
            'c': [1, 2, 3, 4, 5],
            'feature': ['a', 'b', 'b', 'none', 'a+b']
        })
        test_df[['feature']] = test_df[['feature']].astype('category')
        test_df[['c']] = df[['c']].astype('float64')

        new_df = OneHotDecoderTransformer(one_hot_features=['a', 'b'], new_feature='feature').fit_transform(df)

        assert new_df.equals(test_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mi', action='store_true', help='If this flag is set to true, more information about the pipeline progress will be displayed')
    args = parser.parse_args()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)