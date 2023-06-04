from src.features import build_features, sampling_strategies
from src.pipelines.build_pipeline import CustomPipeline
from src.pipelines import pipeline_cleaning
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.tree import DecisionTreeClassifier
from imblearn import FunctionSampler
from imblearn.ensemble import (
    EasyEnsembleClassifier,
    BalancedBaggingClassifier,
    RUSBoostClassifier,
)
from category_encoders.binary import BinaryEncoder
from lightgbm import LGBMClassifier

from src.pipelines.pipeline_cleaning import OutlierRemover


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

    add_outlier_handling(
        custom_pipeline=custom_pipeline,
        outlier_handling_func=pipeline_cleaning.OutlierRemover(
            cat_threshold=0, zscore_threshold=4
        ).handle_outliers,
    )

    # additional feature selection by removing certain columns
    add_remove_feature_transformer(custom_pipeline, ["age"])

    # discretize numerical features
    add_kbinsdiscretizer(custom_pipeline, number_of_bins=2)

    # add encoder and scaler
    add_binary_encoder_and_minmaxscaler(custom_pipeline)

    # add estimator
    apply_tuned_lgbm_classifier(custom_pipeline)


def add_remove_feature_transformer(custom_pipeline: CustomPipeline, features_to_remove):
    feature_remover = build_features.RemoveFeatureTransformer(features_to_remove)
    custom_pipeline.add_new_step(feature_remover, "feature_remover")


def add_outlier_handling(custom_pipeline: CustomPipeline, outlier_handling_func):
    outlier_handler = FunctionSampler(func=outlier_handling_func, validate=False)
    custom_pipeline.add_new_step_at_position(outlier_handler, "outlier_handler", 0)


def add_resampling(custom_pipeline: CustomPipeline, resampling_func):
    resampler = FunctionSampler(func=resampling_func, validate=False)
    custom_pipeline.add_new_step(resampler, "resampler")


def add_kbinsdiscretizer(custom_pipeline: CustomPipeline, number_of_bins: int):
    discretizer = KBinsDiscretizer(
        n_bins=number_of_bins, strategy="uniform", encode="ordinal"
    )
    # apply on all columns
    trans = build_features.CustomColumnTransformer(
        [
            (
                "bins",
                discretizer,
                make_column_selector(
                    pattern="(?!geo_level_1.*)^.*",  # ignores geo-level-1 features
                    dtype_exclude=["category", "object"],
                ),
            ),
            (
                "dummy_numerical",
                build_features.DummyTransformer(),
                make_column_selector(
                    pattern="geo_level_1.*",  # apply to geo-level-1 features only
                    dtype_exclude=["category", "object"],
                ),
            ),  # necessary to keep feature names
            (
                "dummy_categorical",
                build_features.DummyTransformer(),
                make_column_selector(dtype_include=["category", "object"]),
            ),  # necessary to keep feature names
        ],
        remainder="passthrough",
    )
    custom_pipeline.add_new_step(trans, "discretizer")


def add_binary_encoder_and_minmaxscaler(custom_pipeline: CustomPipeline):
    # encodes categorical features
    encoder = BinaryEncoder()

    # scales numerical features
    scaler = MinMaxScaler()

    trans = build_features.CustomColumnTransformer(
        [
            (
                "encoder",
                encoder,
                make_column_selector(dtype_include=["category", "object"]),
            ),
            (
                "scaler",
                scaler,
                make_column_selector(dtype_exclude=["category", "object"]),
            ),
        ],
        remainder="passthrough",
    )
    custom_pipeline.add_new_step(trans, "encoder_and_scaler")


def apply_knn_classifier(custom_pipeline: CustomPipeline, k: int, w, p: int):
    customEstimator = KNeighborsClassifier(n_neighbors=k, weights=w, p=p)
    custom_pipeline.change_estimator(customEstimator)


def apply_default_knn_classifier(custom_pipeline: CustomPipeline):
    customEstimator = KNeighborsClassifier()
    custom_pipeline.change_estimator(customEstimator)


def apply_tuned_knn_classifier(custom_pipeline: CustomPipeline):
    customEstimator = KNeighborsClassifier(
        n_neighbors=9,
        p=1,
        weights='uniform'
    )
    custom_pipeline.change_estimator(customEstimator)


def apply_lgbm_classifier(custom_pipeline: CustomPipeline):
    customEstimator = LGBMClassifier()
    custom_pipeline.change_estimator(new_estimator=customEstimator)


def apply_partly_tuned_lgbm_classifier(custom_pipeline: CustomPipeline):
    customEstimator = LGBMClassifier(
        learning_rate=0.3,
        max_depth=-1,
        n_estimators=400
    )
    custom_pipeline.change_estimator(new_estimator=customEstimator)


def apply_tuned_lgbm_classifier(custom_pipeline: CustomPipeline):
    customEstimator = LGBMClassifier(
        learning_rate=0.3,
        max_depth=-1,
        n_estimators=400,
        metric='multi_error',
        lambda_l1=0.5,
        bagging_fraction=0.7,
        num_leaves=50
    )
    custom_pipeline.change_estimator(new_estimator=customEstimator)


def apply_randomforest_classifier(custom_pipeline: CustomPipeline):
    customEstimator = RandomForestClassifier()
    custom_pipeline.change_estimator(new_estimator=customEstimator)


def apply_tuned_randomforest_classifier(custom_pipeline: CustomPipeline):
    customEstimator = RandomForestClassifier(
        bootstrap=False,
        max_depth=25,
        max_features='auto',
        min_samples_leaf=4,
        min_samples_split=10,
        n_estimators=500
    )
    custom_pipeline.change_estimator(new_estimator=customEstimator)


def apply_balancedbaggingclassifier(custom_pipeline: CustomPipeline):
    model = BalancedBaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        sampling_strategy="auto",
        replacement=False,
        random_state=42,
    )
