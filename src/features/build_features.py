import pandas as pd
import numpy as np
import tqdm as tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders.ordinal import OrdinalEncoder
from pandas.api.types import is_numeric_dtype

def create_one_hot_encoding(df):
    """
    Transforms a dataframe consisting of binary categorical feature into a
    valid one-hot encoding by creating new features for rows that have
    multiple occurrences of 1.

    Example:
        a b      a b a+b
        1 0      1 0 0
        0 1  ->  0 1 0
        1 1      0 0 1

    :param df: a dataframe of binary categorical features
    :return: the encoded dataframe
    """
    # check whether df only contains 0 and 1
    if not df.isin([0, 1]).all(None):
        raise Exception('One or more features is not binary categorical.')

    df_encoded = df.copy()

    # only iterate over rows with multiple occurrences of 1
    for index, row in df[df.sum(axis=1) >= 2].iterrows():
        new_feature = '+'.join(df.columns[row.isin([1])].tolist())

        if new_feature not in df_encoded.columns:
            df_encoded[new_feature] = 0

        df_encoded.loc[index, df_encoded.columns] = 0
        df_encoded.loc[index, new_feature] = 1

    return df_encoded


def is_one_hot_encoded(df):
    """
    Check whether the submitted dataframe is one-hot encoded or not. Each row
    of a one-hot encoded dataframe has to contain exactly one occurence of 1,
    while the remaining entries are 0.

    :param df: the dataframe to be examined
    :return: whether the dataframe is one-hot encoded or not
    """

    # check whether df only contains 0 and 1
    if not df.isin([0, 1]).all(None):
        return False

    # check whether 1 occurs exactly once per row
    row_sums = df.sum(axis=1)
    return row_sums.isin([1]).all()


def revert_one_hot_encoding(df):
    """
    Reverts a one-hot encoded dataframe by returning the feature with the
    highest value for each row. In a one-hot encoded dataframe this is the
    feature with the value 1, while the remaining features have the value 0.

    :param df: the dataframe to be decoded
    :return: the decoded dataframe
    """
    return df.idxmax(axis=1)


def find_value_indices(df, column_name, value):
    """
    Finds the index of a specific value in a Pandas DataFrame and returns a list of indices where the value appears.
    """
    indices = df.index[df[column_name] == value].tolist()
    return indices


# threshold between 0 and 1
def check_dataframe_for_imbalanced_features(df, threshold):
    imbalancedFeatures = []
    for col in df.columns:
        if df[col].dtype == 'category': 
            numberOfrows = df[col].count()
            countedValues = df[col].value_counts()
            for val in countedValues:
                if val > numberOfrows * threshold:
                    imbalancedFeatures.append(col)
    return imbalancedFeatures   


def remove_rows_by_integer_index(df, integerList):
    idx = np.ones(len(df.index), dtype=bool)
    for value in integerList:
        idx[value] = False

    return df.iloc[idx]     


class OneHotDecoderTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms a dataframe into one-hot encoding and then decodes it to get
    a categorical feature. The features referenced in 'one_hot_features' are
    examined and the resulting decoded feature has the name defined in
    'new_feature'.

    Parameters
    ----------
    one_hot_features : list of str, int or float
        List of strings, integers or floats.

    new_feature : str, int of float
        String, integer or float.

    default : str, default='none'
        Default value if a row only has 0s.

    Attributes
    ----------
    one_hot_features : list of str, int or float
        List of strings, integers or floats.

    new_feature : str, int of float
        String, integer or float.

    default : str, default='none'
        Default value if a row only has 0s.

    """

    def __init__(self, one_hot_features, new_feature, default='none'):
        self.one_hot_features = one_hot_features
        self.new_feature = new_feature
        self.default = default

    def fit(self, X, y=None):
        """
        Returns this transformer object.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : DropFeatureTransformer
            This object.
        """
        return self

    def transform(self, X):
        """
        Transforms the dataframe given by the columns whose names occur in the
        class attribute 'one_hot_features' into a valid one-hot encoding and
        then reverts this encoding. The resulting categorical feature has the
        name defined in 'new_feature'.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        df = X[self.one_hot_features]

        # check whether df only contains 0 and 1
        if not df.isin([0, 1]).all(None):
            raise Exception('One or more features is not binary categorical.')

        df_encoded = df.copy()
        df = df.astype('int')

        # create a new feature for rows with no occurrences
        rows_without_occurrences = (df.sum(axis=1) == 0)
        if len(rows_without_occurrences) != 0:
            df_encoded[rows_without_occurrences] = 0
            df_encoded[self.default] = 0
            df_encoded[self.default][rows_without_occurrences] = 1

        # only iterate over rows with multiple occurrences of 1
        for index, row in df[df.sum(axis=1) >= 2].iterrows():
            new_feature = '+'.join(df.columns[row.isin([1])].tolist())

            if new_feature not in df_encoded.columns:
                df_encoded[new_feature] = 0

            df_encoded.loc[index, df_encoded.columns] = 0
            df_encoded.loc[index, new_feature] = 1

        X_new = X.copy()
        X_new = X_new.drop(columns=self.one_hot_features)
        X_new[self.new_feature] = df_encoded.idxmax(axis=1)
        X_new[self.new_feature] = X_new[self.new_feature].astype('category')

        return X_new

class RemoveFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that removes features from a given dataset. This is done
    according to their name and a submitted list of features to remove.

    Parameters
    ----------
    features_to_drop : list of str, int or float, default=None
        List of strings, integers or floats.

    Attributes
    ----------
    features_to_drop : list of str, int or float, default=None
        List of strings, integers or floats.

    """

    def __init__(self, features_to_drop=None):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        """
        Returns this transformer object.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : DropFeatureTransformer
            This object.
        """
        return self

    def transform(self, X):
        """
        Removes columns with names that occur in the class attribute 'features_to_drop'.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        if not self.features_to_drop:
            return X

        features_to_keep = pd.DataFrame(X).columns.tolist()

        for feature in self.features_to_drop:
            if feature in features_to_keep:
                features_to_keep.remove(feature)

        return X[features_to_keep]


class DummyTransformer(BaseEstimator, TransformerMixin):
    """
    Dummy transformer that does not alter the data.
    """

    def fit(self, X, y=None):
        """
        Returns this transformer object.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : DropFeatureTransformer
            This object.
        """
        return self

    def transform(self, X):
        """
        Returns the input samples without altering them.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X : ndarray array of shape (n_samples, n_features)
            Input samples.
        """
        return X


class CustomColumnTransformer(ColumnTransformer):
    """
    ColumnTransformer variant that is tailored towards pandas DataFrames. Fixes
    issues regarding column names being lost.
    """


    def get_feature_names(column_transformer):
        """Get feature names from all transformers.
        Returns
        -------
        feature_names : list of strings
            Names of the features produced by transform.
        """

        # Remove the internal helper function
        # check_is_fitted(column_transformer)

        # Turn loopkup into function for better handling with pipeline later
        def get_names(trans):
            # >> Original get_feature_names() method
            if trans == 'drop' or (
                    hasattr(column, '__len__') and not len(column)):
                return []
            if trans == 'passthrough':
                if hasattr(column_transformer, '_df_columns'):
                    if ((not isinstance(column, slice))
                            and all(isinstance(col, str) for col in column)):
                        return column
                    else:
                        return column_transformer._df_columns[column]
                else:
                    indices = np.arange(column_transformer._n_features)
                    return ['x%d' % i for i in indices[column]]
            if not hasattr(trans, 'get_feature_names'):
                # >>> Change: Return input column names if no method avaiable
                # Turn error into a warning
                #             warnings.warn("Transformer %s (type %s) does not "
                #                                  "provide get_feature_names. "
                #                                  "Will return input column names if available"
                #                                  % (str(name), type(trans).__name__))
                # For transformers without a get_features_names method, use the input
                # names to the column transformer
                if column is None:
                    return []
                else:
                    return [  # name + "__" +
                        f for f in column]

            return [  # name + "__" +
                f for f in trans.get_feature_names()]

        ### Start of processing
        feature_names = []

        # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
        if type(column_transformer) == Pipeline:
            l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
        else:
            # For column transformers, follow the original method
            l_transformers = list(column_transformer._iter(fitted=True))

        for name, trans, column, _ in l_transformers:
            if type(trans) == Pipeline:
                # Recursive call on pipeline
                _names = column_transformer.get_feature_names(trans)
                # if pipeline has no transformer that returns names
                if len(_names) == 0:
                    _names = [  # name + "__" +
                        f for f in column]
                feature_names.extend(_names)
            else:
                feature_names.extend(get_names(trans))

        return feature_names

    def transform(self, X):
        indices = X.index.values.tolist()
        original_columns = X.columns.values.tolist()
        X_mat = super().transform(X)
        new_cols = self.get_feature_names()
        new_X = pd.DataFrame(X_mat, index=indices, columns=new_cols)
        return new_X

    def fit_transform(self, X, y=None):
        super().fit_transform(X, y)
        return self.transform(X)

class CombineFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, feature1, feature2):
        self.feature1 = feature1
        self.feature2 = feature2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        #if is_numeric_dtype(X[self.feature1]) or is_numeric_dtype(X[self.feature2].dtype):
        #    X[self.feature1] = X[self.feature1].astype(str)
        #    X[self.feature2] = X[self.feature2].astype(str)
        
        X[self.feature1 + ' ' + self.feature2] = X[self.feature1] + X[self.feature2]
        return X