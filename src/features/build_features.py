import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders.ordinal import OrdinalEncoder


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

def find_outliers_by_threshold(df, threshold = 0.02, displayInfo = False):
    """   
    Creates a list containing outliers for each feature of the dataframe.  

    Args:
        df: the dataframe
        threshold (integer): the threshold used to determine wether a feature value should be removed
    return: Returns a list containing the values for each feature that should be dropped based on the treshold
    """
    threshold = 1 - threshold
    if displayInfo:
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
    outliers = {}
    featuresWithNoOutliers = []
    for feature, counts in value_counts.items():
        
        # get number of values for feature
        number_of_feature_values = df[feature].count()
        
        # for each value in value_counts calculate the relative percentage
        percentage_counts[feature] = {}
        outliers[feature] = []
        cumulativePercentage = 0
        
        hasOutliers = False
        for val, countOfValues in counts.items():
            percentage_counts[feature][val] = countOfValues / number_of_feature_values            
            if(cumulativePercentage > threshold):
                outliers[feature].append(val)
                hasOutliers = True
            cumulativePercentage += percentage_counts[feature][val]
        if not(hasOutliers):
            featuresWithNoOutliers.append(feature)
    
    if displayInfo:
        print(f'Features that have no outliers are: {featuresWithNoOutliers}')
        print('Features with outliers:')
        for feature, values in outliers.items():
            if feature in featuresWithNoOutliers:
                continue
            
            print(f'Outliers for feature: {feature}')
            print(f'    A total of {len(values)} outliers have been found')
            maxDisplay = 10
            for i in range(0, maxDisplay):
                if i >= len(values):                    
                    break
                print(f'    {i}: {values[i]}')
                if i == maxDisplay -1 and len(values) >= maxDisplay:
                    print(f'.. {len(values) - maxDisplay} more outliers were found')
    
    return outliers   

#create a function to find outliers using IQR
def find_outliers_IQR(list):

   q1=list.quantile(0.25)
   q3=list.quantile(0.75)
   IQR=q3-q1
   outliers = list[((list<(q1-1.5*IQR)) | (list>(q3+1.5*IQR)))]

   return outliers     

def find_outliers_IQR_asindizes(list):
    q1 = list.quantile(0.25)
    q3 = list.quantile(0.75)
    IQR = q3 - q1
    mask = ((list < (q1 - 1.5 * IQR)) | (list > (q3 + 1.5 * IQR)))
    outlier_indices = mask[mask == True].index

    return outlier_indices

def find_outliers_Zscore(list, threshold=3):
     # Calculate the mean and standard deviation of the array
    mean = np.mean(list)
    std_dev = np.std(list)

    # Calculate the z-score for each element in the array
    z_scores = (list - mean) / std_dev

    # Create a boolean mask for elements that have a z-score greater than the threshold
    mask = np.abs(z_scores) > threshold

    # Create a Pandas series of values for the elements that have a z-score greater than the threshold
    outlier_values = pd.Series(list[mask])

    return outlier_values

def find_zscore_outliers_asindizes(list, threshold=3):

    # Calculate the mean and standard deviation of the array
    mean = np.mean(list)
    std_dev = np.std(list)

    # Calculate the z-score for each element in the array
    z_scores = (list - mean) / std_dev

    # Create a boolean mask for elements that have a z-score greater than the threshold
    mask = np.abs(z_scores) > threshold

    # Create a Pandas series of indices with the boolean mask as the index values
    outlier_indices = pd.Series(range(len(list)))[mask]

    return outlier_indices


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
