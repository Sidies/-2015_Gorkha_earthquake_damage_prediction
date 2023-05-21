from src.features import build_features
from src.pipelines.build_pipeline import CustomPipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier, RUSBoostClassifier
from category_encoders.binary import BinaryEncoder
from lightgbm import LGBMClassifier

def add_best_steps(custom_pipeline: CustomPipeline):
    """
    Returns a list of tuples containing transformation steps to be applied in a Pipeline.
    
    If no custom estimator is provided, the function uses the KNeighborsClassifier with 9 neighbors as the default estimator.

    Parameters:
    customEstimator: estimator object implementing 'fit', default=None
        The estimator to use. If not provided a default classifier will be used.

    Returns:
    steps: list of tuples
        Each tuple contains a step name and an instance of the transformer or estimator to be applied in the Pipeline.
    """
    
    # additional feature selection by removing certain columns
    add_remove_feature_transformer(custom_pipeline, ['age'])
    
    # discretize numerical features
    add_kbinsdiscretizer(custom_pipeline, number_of_bins=2)

    # add encoder and scaler
    add_binary_encoder_and_minmaxscaler(custom_pipeline)

    # add estimator
    apply_knn_classifier(custom_pipeline, 9)
    
    
def add_remove_feature_transformer(custom_pipeline: CustomPipeline, features_to_remove):
    feature_remover = build_features.RemoveFeatureTransformer(features_to_remove)
    custom_pipeline.add_new_step(feature_remover, 'feature_remover')


def add_kbinsdiscretizer(custom_pipeline: CustomPipeline, number_of_bins: int):
    discretizer = KBinsDiscretizer(n_bins=number_of_bins, strategy='uniform', encode='ordinal')
    # apply on all columns
    trans = build_features.CustomColumnTransformer([
            ('bins', discretizer, make_column_selector(dtype_exclude=['category', 'object'])),
            ('dummy', build_features.DummyTransformer(), make_column_selector(dtype_include=['category', 'object'])) # necessary to keep feature names
        ], remainder='passthrough')
    custom_pipeline.add_new_step(trans, 'discretizer')
    
    
def add_binary_encoder_and_minmaxscaler(custom_pipeline: CustomPipeline):
    # encodes categorical features
    encoder = BinaryEncoder()

    # scales numerical features
    scaler = MinMaxScaler()
    trans = build_features.CustomColumnTransformer([
            ('encoder', encoder, make_column_selector(dtype_include=['category', 'object'])),
            ('scaler', scaler, make_column_selector(dtype_exclude=['category', 'object']))
        ], remainder='passthrough')
    custom_pipeline.add_new_step(trans, 'encoder_and_scaler')
    
    
def add_randomsampling(custom_pipeline: CustomPipeline, oversampling_strategy='auto', undersampling_strategy='auto'):
    # Define oversampling strategy
    over = RandomOverSampler(sampling_strategy=oversampling_strategy, random_state=42)
    custom_pipeline.add_new_step(over, 'oversampling')
    # Define undersampling strategy
    under = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)
    custom_pipeline.add_new_step(under, 'undersampling')
    
    
def apply_knn_classifier(custom_pipeline: CustomPipeline, k: int):
    customEstimator = KNeighborsClassifier(n_neighbors=k)
    custom_pipeline.change_estimator(customEstimator)
    
    
def apply_lgbm_classifier(custom_pipeline: CustomPipeline):
    customEstimator = LGBMClassifier()
    custom_pipeline.change_estimator(new_estimator=customEstimator)
    
def apply_randomforest_classifier(custom_pipeline: CustomPipeline):
    customEstimator = RandomForestClassifier()
    custom_pipeline.change_estimator(new_estimator=customEstimator)
    
def apply_balancedbaggingclassifier(custom_pipeline: CustomPipeline):
    model = BalancedBaggingClassifier(
        base_estimator=DecisionTreeClassifier(), 
        sampling_strategy='auto', 
        replacement=False, 
        random_state=42)
