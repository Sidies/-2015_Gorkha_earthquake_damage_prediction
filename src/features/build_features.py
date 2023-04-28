import pandas as pd
import numpy as np

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

def find_outliers_by_threshold(df, threshold = 0.98, displayInfo = False):
    """   
    Creates a list containing outliers for each feature of the dataframe.  

    Args:
        df: the dataframe
        threshold (integer): the threshold used to determine wether a feature value should be removed
    return: Returns a list containing the values for each feature that should be dropped based on the treshold
    """
    print(f"Start: filtering out outliers with threshold {threshold}")
    
    if threshold > 1.0 or threshold < 0:
        print('Error: The threshold for filtering values in features has to be between 1 and 0.')
        return 
    
    # count the number of occurrences of each value for each feature
    value_counts = {}
    for feature in df.columns:
        value_counts[feature] = df[feature].value_counts().sort_values(ascending=False)   
    
    # convert the values to the relative percentage
    percentage_counts = {}
    values_below_threshold = {}
    for feature, counts in value_counts.items():
        # get number of values for feature
        number_of_feature_values = df[feature].count()
        
        # for each value in value_counts calculate the relative percentage
        # and 
        percentage_counts[feature] = {}
        values_below_threshold[feature] = []
        cumulativePercentage = 0
        for val, countOfValues in counts.items():
            percentage_counts[feature][val] = countOfValues / number_of_feature_values
            cumulativePercentage += percentage_counts[feature][val]
            if(cumulativePercentage > threshold):
                values_below_threshold[feature].append(val)
    
    return values_below_threshold   
    if displayInfo:
        for feature, values in values_below_threshold.items():
            print(f'feature name: {feature}')
            maxDisplay = 10
            for i in range(0, maxDisplay):
                if i >= len(values):                    
                    break
                print(f'    {i}: {values[i]}')
                if i == maxDisplay -1 and len(values) >= maxDisplay:
                    print(f'.. {len(values) - maxDisplay} more outliers were found')
    
    return values_below_threshold   


# threshold between 0 and 1
def check_dataframe_for_imbalanced_features(df, threshold=0.9):
    imbalancedFeatures = []
    for col in df.columns:
        if df[col].dtype != object or df[col].dtype.name != 'category':
            continue
        numberOfrows = df[col].count()
        countedValues = df[col].value_counts()
        for val in countedValues:
            if val > numberOfrows * threshold:
                imbalancedFeatures.append(col)
    return imbalancedFeatures        
             

#create a function to find outliers using IQR
def find_outliers_IQR(df):

   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

   return outliers     

def find_outlier_ZScore(df):
    
    threshold=3
    mean_1 = np.mean(df)
    std_1 =np.std(df)
    
    outliers = []
    for y in df:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


class DropRowsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rows_to_drop):
        self.rows_to_drop = rows_to_drop
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        for column, row_names in self.rows_to_drop.items():            
            indizesToDrop = []
            for value in row_names:
                indizesToDrop.append(X[column].index[X[column] == value][0])
            X[column] = X[column].drop(labels=indizesToDrop)
        X = X.dropna()
        return X

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
