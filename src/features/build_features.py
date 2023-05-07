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

def find_outliers_by_threshold(df, threshold = 0.02, displayInfo = False):
    """   
    Creates a list containing outliers for each feature of the dataframe.  

    Args:
        df: the dataframe
        threshold (integer): the threshold used to determine wether a feature value should be removed
    return: Returns a list containing the values for each feature that should be dropped based on the treshold
    """
    threshold = 1 - threshold
    #if displayInfo:
    #   print(f"Start: filtering out outliers with threshold {threshold}")
    
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

def find_outliers_by_threshold_as_index(df, threshold = 0.02):
    threshold = 1 - threshold
    
    if threshold > 1.0 or threshold < 0:
        print('Error: The threshold for filtering values in features has to be between 1 and 0.')
        return 
    
    # count the number of occurrences of each value for each feature
    value_counts = {}
    for feature in df.columns:
        value_counts[feature] = df[feature].value_counts().sort_values(ascending=False)   
    
    # convert the values to the relative percentage
    percentage_counts = {}
    outliersAsIndex = {}
    featuresWithNoOutliers = []
    for feature, counts in value_counts.items():
        
        # get number of values for feature
        number_of_feature_values = df[feature].count()
        
        # for each value in value_counts calculate the relative percentage
        percentage_counts[feature] = {}
        outliersAsIndex[feature] = []
        cumulativePercentage = 0
        
        hasOutliers = False
        for val, countOfValues in counts.items():
            percentage_counts[feature][val] = countOfValues / number_of_feature_values            
            if(cumulativePercentage > threshold):
                outliersAsIndex[feature].append(df.index[df[feature] == val])
                hasOutliers = True
            cumulativePercentage += percentage_counts[feature][val]
        if not(hasOutliers):
            featuresWithNoOutliers.append(feature)
            
    return outliersAsIndex


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

def remove_outliers_from_dataframes(df, numerical_columns, categorical_columns, threshold = 0.2, minfo = False):
    
    ######################
    # Remove numerical outliers
    ######################
    iqr_indizes = {}
    z_indizes = {}
    previousSize = len(df)
    #print(f'Previous dataframe size {len(df)}')
    for feature in df[numerical_columns]:
        z_indizes[feature] = find_zscore_outliers_asindizes(df[feature], 2)
        iqr_indizes[feature] = find_outliers_IQR_asindizes(df[feature])
        
        # convert the lists to sets for easy intersection
        zscore_outliers_set = set(z_indizes[feature])
        iqr_outliers_set = set(iqr_indizes[feature])

        # find the common outliers
        common_outliers = zscore_outliers_set.intersection(iqr_outliers_set)
        amountToDrop = len(common_outliers)
        expectedAmount = len(df[feature]) - amountToDrop
        #print(f'{amountToDrop} should be dropped')

        # assuming your data is in a pandas DataFrame named 'df'
        # remove the common outliers from the DataFrame
        df = df.drop(common_outliers)
        df = df.reset_index(drop=True)
        #print(f'Expected amount: {expectedAmount} and current amount: {len(df[feature])}')
    #newSize = len(df)
    #print(f'New dataframe size {len(df)}')
    #print(f'A total of {previousSize - newSize} rows have been dropped')
    
    ######################
    # Remove categorical outliers
    ######################
    
    cat_outliers = find_outliers_by_threshold(df[categorical_columns], threshold, False)
    previousSize = len(df)
    indizesToRemove = {}
    #print(f'Previous dataframe size {len(df)}')
    for feature in df[categorical_columns]:
        
        # calculate the index that should be dropped
        indizesToRemove[feature] = [] # Initialize an empty list for each feature
        for value in cat_outliers[feature]:        
            indizesToRemove[feature].extend(find_value_indices(df[categorical_columns],feature, value))
        
        amountToDrop = len(indizesToRemove[feature])
        expectedAmount = len(df[feature]) - amountToDrop
        #print(f'{amountToDrop} should be dropped')

        # remove the common outliers from the DataFrame
        df = df.drop(indizesToRemove[feature])
        df = df.reset_index(drop=True)
        #print(f'Expected amount: {expectedAmount} and current amount: {len(df[feature])}')
    newSize = len(df)
    #print(f'New dataframe size {len(df)}')
    #print(f'A total of {previousSize - newSize} rows have been dropped')
    
    return df

def get_outlier_rows_as_index(df, numerical_columns, categorical_columns, threshold = 0.2, minfo = False):
    '''
    Returns a list of outliers with an integer list representating the index position
    '''
    all_outlier_row_indizes = []
    ######################
    # Find numerical outliers
    ######################
    iqr_indizes = {}
    z_indizes = {}
    for feature in df[numerical_columns]:
        z_indizes[feature] = find_zscore_outliers_asindizes(df[feature], 2)
        iqr_indizes[feature] = find_outliers_IQR_asindizes(df[feature])
        
        # convert the lists to sets for easy intersection
        zscore_outliers_set = set(z_indizes[feature])
        iqr_outliers_set = set(iqr_indizes[feature])

        # find the common outliers
        common_outliers = zscore_outliers_set.intersection(iqr_outliers_set)
        all_outlier_row_indizes.extend(common_outliers)
    
    ######################
    # Find categorical outliers
    ######################
    
    cat_outliers = find_outliers_by_threshold_as_index(df[categorical_columns], threshold)
    for feature in df[categorical_columns]:
        # calculate the index that should be dropped      
        all_outlier_row_indizes.extend(cat_outliers[feature])
      
    return all_outlier_row_indizes

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

        # only interate over rows with no occurences of 1
        for index, row in df[df.sum(axis=1) == 0].iterrows():
            new_feature = self.default

            if new_feature not in df_encoded.columns:
                df_encoded[new_feature] = 0

            df_encoded.loc[index, df_encoded.columns] = 0
            df_encoded.loc[index, new_feature] = 1

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
        
        if is_numeric_dtype(X[self.feature1]) or is_numeric_dtype(X[self.feature2].dtype):
            X[self.feature1] = X[self.feature1].astype(str)
            X[self.feature2] = X[self.feature2].astype(str)
        
        X[self.feature1 + ' ' + self.feature2] = X[self.feature1] + ' ' + X[self.feature2]
        return X