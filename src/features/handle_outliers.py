import pandas as pd
import numpy as np
from src.features import build_features


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
    """
    Identify the indices of outliers in a DataFrame, using a given threshold.

    Args:
        df (pandas.DataFrame): DataFrame to analyze.
        threshold (float): Threshold to identify outliers.

    Returns:
        dict: Dictionary containing feature names as keys and lists of outlier indices as values.
    """
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


def find_outliers_IQR(list):
    """
    Identify the outliers of a list using the Interquartile Range (IQR).

    Args:
        list (pandas.Series): Series of numerical data to analyze.

    Returns:
        pandas.Series: Series of outliers.
    """
    q1=list.quantile(0.25)
    q3=list.quantile(0.75)
    IQR=q3-q1
    outliers = list[((list<(q1-1.5*IQR)) | (list>(q3+1.5*IQR)))]

    return outliers     


def find_outliers_IQR_asindizes(list):
    """
    Identify the indices of outliers in a list using the Interquartile Range (IQR).

    Args:
        list (pandas.Series): Series of numerical data to analyze.

    Returns:
        pandas.Index: Index of outliers.
    """
    q1 = list.quantile(0.25)
    q3 = list.quantile(0.75)
    IQR = q3 - q1
    mask = ((list < (q1 - 1.5 * IQR)) | (list > (q3 + 1.5 * IQR)))
    outlier_indices = mask[mask == True].index

    return outlier_indices


def find_outliers_Zscore(list, threshold=3):
    """
    Identify the outliers of a list using the Z-Score.

    Args:
        list (pandas.Series): Series of numerical data to analyze.
        threshold (float): Z-Score threshold to identify outliers.

    Returns:
        pandas.Series: Series of outliers.
    """
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
    """
    Identify the indices of outliers in a list using the Z-Score.

    Args:
        list (pandas.Series): Series of numerical data to analyze.
        threshold (float): Z-Score threshold to identify outliers.

    Returns:
        pandas.Index: Index of outliers.
    """
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


def get_outlier_rows_as_index(df, numerical_columns, categorical_columns, threshold = 0.2, minfo = False):
    """
    Get the indices of outlier rows in a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame to analyze.
        numerical_columns (list of str): List of numerical column names.
        categorical_columns (list of str): List of categorical column names.
        threshold (float): Threshold to identify categorical outliers.
        minfo (bool): If True, print information about the analysis process.

    Returns:
        list: List of outlier indices.
    """
    all_outlier_row_indizes = []

    # Find numerical outliers

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
    

    # Find categorical outliers
    
    cat_outliers = find_outliers_by_threshold_as_index(df[categorical_columns], threshold)
    for feature in df[categorical_columns]:
        # calculate the index that should be dropped      
        all_outlier_row_indizes.extend(cat_outliers[feature])
      
    return all_outlier_row_indizes


def remove_outliers_from_dataframes(df, numerical_columns, categorical_columns, threshold = 0.2, minfo = False):
    """
    Remove outliers from a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame to clean.
        numerical_columns (list of str): List of numerical column names.
        categorical_columns (list of str): List of categorical column names.
        threshold (float): Threshold to identify categorical outliers.
        minfo (bool): If True, print information about the cleaning process.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    # Remove numerical outliers

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
    

    # Remove categorical outliers
    
    cat_outliers = find_outliers_by_threshold(df[categorical_columns], threshold, False)
    previousSize = len(df)
    indizesToRemove = {}
    #print(f'Previous dataframe size {len(df)}')
    for feature in df[categorical_columns]:
        
        # calculate the index that should be dropped
        indizesToRemove[feature] = [] # Initialize an empty list for each feature
        for value in cat_outliers[feature]:        
            indizesToRemove[feature].extend(build_features.find_value_indices(df[categorical_columns],feature, value))
        
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


