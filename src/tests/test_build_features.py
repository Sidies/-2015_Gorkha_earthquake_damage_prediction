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
        
    def test_CombineFeatureTransformer(self):
        df = pd.DataFrame({
            'first': ['Dirk', 'Kobe', 'Tim', 'Lebron'],
            'last': ['Nowitzki', 'Bryant', 'Duncan', 'James'],
        })

        test_df = pd.DataFrame({
            'first': ['Dirk', 'Kobe', 'Tim', 'Lebron'],
            'last': ['Nowitzki', 'Bryant', 'Duncan', 'James'],
            'first last': ['Dirk Nowitzki', 'Kobe Bryant', 'Tim Duncan', 'Lebron James'],
        })

        new_df = CombineFeatureTransformer('first', 'last').fit_transform(df)

        assert new_df.equals(test_df), f"Expected the dataframes to match but got: \n {new_df}"
        
        df = pd.DataFrame({
            'a': [1, 0, 0, 0, 1],
            'b': [0, 1, 1, 0, 1],
        })

        test_df = pd.DataFrame({
            'a': [1, 0, 0, 0, 1],
            'b': [0, 1, 1, 0, 1],
            'a b': ['1 0', '0 1', '0 1', '0 0', '1 1'],
        })
        
        new_df = CombineFeatureTransformer('a', 'b').fit_transform(df)

        assert new_df.equals(test_df), f"Expected the dataframes to match but got: \n {new_df} instead of \n {test_df}"
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mi', action='store_true', help='If this flag is set to true, more information about the pipeline progress will be displayed')
    args = parser.parse_args()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)