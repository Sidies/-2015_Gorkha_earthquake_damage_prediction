import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder


def reverse_one_hot_encoding(encoding):
    """
    Funktion, die das One-Hot-Encoding zurücksetzt und eine Liste von Kategorien zurückgibt.
    """
    # ein leeres Set für die einzigartigen Kategorien erstellen
    unique_categories = set()

    # die einzigartigen Kategorien aus der binären Matrix extrahieren
    for binary in encoding:
        unique_categories.add(tuple(binary))

    # eine Liste der Kategorien erstellen, indem jedes Element der binären Matrix mit der entsprechenden Kategorie abgeglichen wird
    categories = []
    for binary in encoding:
        for category in unique_categories:
            if tuple(binary) == category:
                categories.append(list(unique_categories).index(category))

    # die Liste der Kategorien zurückgeben
    return categories


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
