import pandas as pd
from src.features import build_features
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class Sampler:
    def apply_sampling(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        return X_train, y_train
    

class RandomSampler(Sampler):
    def __init__(self, oversampling_strategy='auto', undersampling_strategy='auto'):
        self.oversampling_strategy = oversampling_strategy
        self.undersampling_strategy=undersampling_strategy
    
    def apply_sampling(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        
        # Create an instance of RandomOverSampler for oversampling
        over_sampler = RandomOverSampler(sampling_strategy=self.oversampling_strategy, random_state=42)

        # Create an instance of RandomUnderSampler for undersampling
        under_sampler = RandomUnderSampler(sampling_strategy=self.undersampling_strategy, random_state=42)

        # Apply the resampling on the training set
        X_train_resampled, y_train_resampled = over_sampler.fit_resample(X_train, y_train)
        X_train_resampled, y_train_resampled = under_sampler.fit_resample(X_train_resampled, y_train_resampled)
        
        return X_train_resampled, y_train_resampled
    