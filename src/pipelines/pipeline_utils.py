from src.features import build_features
from sklearn.preprocessing import KBinsDiscretizer
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer, make_column_selector

def get_best_steps(customEstimator=None):
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
    feature_remover = build_features.RemoveFeatureTransformer(['age'])

    # discretize numerical features
    discretizer = KBinsDiscretizer(n_bins=2, strategy='uniform', encode='ordinal')

    # encodes categorical features
    encoder = BinaryEncoder()

    # scales numerical features
    scaler = MinMaxScaler()

    # trains and predicts on the transformed data
    if customEstimator == None:
        #customEstimator = LGBMClassifier()
        customEstimator = KNeighborsClassifier(n_neighbors=9)

    return [
        ('feature_remover', feature_remover),
        ('discretizer', build_features.CustomColumnTransformer([
            ('bins', discretizer, make_column_selector(dtype_exclude=['category', 'object'])),
            ('dummy', build_features.DummyTransformer(), make_column_selector(dtype_include=['category', 'object'])) # necessary to keep feature names
        ], remainder='passthrough')),
        ('encoder_and_scaler', build_features.CustomColumnTransformer([
            ('encoder', encoder, make_column_selector(dtype_include=['category', 'object'])),
            ('scaler', scaler, make_column_selector(dtype_exclude=['category', 'object']))
        ], remainder='passthrough')),
        ('estimator', customEstimator)
    ]