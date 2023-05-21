import pandas as pd
import src.data.configuration as config
from src.features import handle_outliers
from src.features import build_features

class OutlierHandler:
    
    def handle_outliers(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        return X_train, y_train
   
class OutlierRemover(OutlierHandler): 
    def __init__(self, cat_threshold=0.2, zscore_threshold=3):
        self.cat_threshold = cat_threshold
        self.zscore_value=zscore_threshold
    
    def handle_outliers(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        # remove the has_secondary_use and has_superstructure columns to not run analysis on them
        new_categorical_columns = list(
            set(config.categorical_columns) - set(config.has_superstructure_columns) - set(config.has_secondary_use_columns))

        # rows we found to contain outliers which can therefore be dropped
        row_indizes_to_remove = handle_outliers.get_outlier_rows_as_index(X_train, config.numerical_columns,
                                                                            new_categorical_columns, 
                                                                            self.cat_threshold, 
                                                                            self.zscore_value)

        X_train = build_features.remove_rows_by_integer_index(X_train, row_indizes_to_remove)
        y_train = build_features.remove_rows_by_integer_index(y_train, row_indizes_to_remove)

        if len(X_train) != len(y_train):
            print('Error: X_train is not equal length to y_train!')
        
        return X_train, y_train