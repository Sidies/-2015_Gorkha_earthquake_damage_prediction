import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import chi2_contingency, f_oneway


def get_verbose_value_counts(series):
    """
    Creates a dataframe containing absolute and relative value counts for the
    submitted series.

    :param series: the series to be examined
    :return: a dataframe containing absolute and relative value counts
    """
    value_counts = series.value_counts()
    total = len(series)
    
    dataframe = pd.DataFrame(columns=['value', 'count (abs)', 'count (rel)'])
    for value, count in zip(value_counts.index, value_counts.values):
        dataframe = pd.concat([
            dataframe, 
            pd.DataFrame([{
                'value': value,
                'count (abs)': count,
                'count (rel)': '{0:.2%}'.format(count / total)
            }])
        ], ignore_index=True)
        
    return dataframe
    
    
def get_verbose_sums(dataframe, total=None):
    """
    Creates a dataframe containing absolute and relative sums per feature for
    the submitted dataframe.

    :param dataframe: the dataframe to be examined
    :param total: the total sum, by default the sum of all values
    :return: a dataframe containing absolute and relative sums per feature
    """
    if not total:
        total = dataframe.values.sum()
    
    sums = dataframe.sum().sort_values(ascending=False)
    
    dataframe = pd.DataFrame(columns=['feature', 'sum (abs)', 'sum (rel)'])
    for feature, feature_sum in zip(sums.index, sums.values):
        dataframe = pd.concat([
            dataframe, 
            pd.DataFrame([{
                'feature': feature,
                'sum (abs)': feature_sum,
                'sum (rel)': '{0:.2%}'.format(feature_sum / total)
            }])
        ], ignore_index=True)
        
    return dataframe
    
    
def get_verbose_occurences(dataframe):
    """
    Creates a dataframe containing absolute and relative occurences of 1 per
    feature for the submitted dataframe, which has to consist of binary 
    categorical features.
    
    :param dataframe: a dataframe consisting of binary categorical features
    :return: a dataframe of absolute and relative occurences of 1 per feature
    """
    dataframe = get_verbose_sums(dataframe, total=len(dataframe))
    dataframe.columns = ['feature', 'occur. (abs)', 'occur. (rel)']
    return dataframe
    
    
def get_verbose_correlations(dataframe, first_feature_list, second_feature_list):
    dataframe = pd.DataFrame(
        data=OrdinalEncoder().fit_transform(dataframe),
        index=dataframe.index,
        columns=dataframe.columns
    )
    
    first_feature_list = first_feature_list.copy()
    second_feature_list = second_feature_list.copy()
    
    correlations = pd.DataFrame()
    for first_feature in first_feature_list:
    
        if first_feature in second_feature_list:
            second_feature_list.remove(first_feature)
            
        for second_feature in second_feature_list:
        
            corr = dataframe[first_feature].corr(dataframe[second_feature])
            correlations = pd.concat([
                correlations,
                pd.DataFrame({
                    'feature_1': [first_feature],
                    'feature_2': [second_feature],
                    'corr.': [corr],
                    'corr. (abs.)': [abs(corr)]
                })
            ], ignore_index=True)
            
    correlations = correlations.sort_values(
        by='corr. (abs.)',
        ascending=False, 
        ignore_index=True
    )
            
    return correlations


def get_verbose_chi_square_statistics(dataframe, first_feature_list, second_feature_list):
    dataframe = pd.DataFrame(
        data=OrdinalEncoder().fit_transform(dataframe),
        index=dataframe.index,
        columns=dataframe.columns
    )
    
    first_feature_list = first_feature_list.copy()
    second_feature_list = second_feature_list.copy()
    
    statistics = pd.DataFrame()
    for first_feature in first_feature_list:
    
        if first_feature in second_feature_list:
            second_feature_list.remove(first_feature)
            
        for second_feature in second_feature_list:
        
            cross_tab = pd.crosstab(
                dataframe[first_feature], 
                dataframe[second_feature]
            )
            chi2, pval, dof, exp_freq = chi2_contingency(cross_tab)
            
            statistics = pd.concat([
                statistics,
                pd.DataFrame({
                    'feature_1': [first_feature],
                    'feature_2': [second_feature],
                    'statistic': [chi2],
                    'p-value': [pval]
                })
            ], ignore_index=True)
            
    statistics = statistics.sort_values(
        by='statistic',
        ascending=False, 
        ignore_index=True
    )
            
    return statistics
    
    
def get_verbose_anova_statistics(dataframe, categorical_feature_list, numerical_feature_list):
    dataframe = pd.DataFrame(
        data=OrdinalEncoder().fit_transform(dataframe),
        index=dataframe.index,
        columns=dataframe.columns
    )
    
    categorical_feature_list = categorical_feature_list.copy()
    numerical_feature_list = numerical_feature_list.copy()
    
    statistics = pd.DataFrame()
    for categorical_feature in categorical_feature_list:
    
        if categorical_feature in numerical_feature_list:
            numerical_feature_list.remove(categorical_feature)
            
        for numerical_feature in numerical_feature_list:
            
            df = dataframe[[categorical_feature, numerical_feature]]
            df = df.groupby(categorical_feature)[numerical_feature].apply(list)
            statistic, pvalue = f_oneway(*df)
            
            statistics = pd.concat([
                statistics,
                pd.DataFrame({
                    'feature_1': [categorical_feature],
                    'feature_2': [numerical_feature],
                    'statistic': [statistic],
                    'p-value': [pvalue]
                })
            ], ignore_index=True)
            
    statistics = statistics.sort_values(
        by='statistic',
        ascending=False, 
        ignore_index=True
    )
            
    return statistics
