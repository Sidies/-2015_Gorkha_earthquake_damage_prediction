import numpy as np
import pandas as pd
import tqdm as tqdm


from joblib import dump
from pathlib import Path
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier

from category_encoders.binary import BinaryEncoder
from category_encoders.glmm import GLMMEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.target_encoder import TargetEncoder

from lightgbm import LGBMClassifier

import os

from src.features.build_features import CustomColumnTransformer, DummyTransformer, OneHotDecoderTransformer, \
    RemoveFeatureTransformer
from src.features import build_features
from src.visualization.visualize import get_verbose_correlations
from src.data import configuration as config


def get_best_steps():
    # additional feature selection by removing certain columns
    feature_remover = RemoveFeatureTransformer([])

    # feature engineering
    feature_engineering = DummyTransformer()

    # decode onehot-encoded secondary use features
    onehot_decoder_secondary_use = OneHotDecoderTransformer(
        one_hot_features=[
            'has_secondary_use_agriculture',
            'has_secondary_use_hotel',
            'has_secondary_use_rental',
            'has_secondary_use_institution',
            'has_secondary_use_school',
            'has_secondary_use_industry',
            'has_secondary_use_health_post',
            'has_secondary_use_gov_office',
            'has_secondary_use_use_police',
            'has_secondary_use_other'
        ],
        new_feature='secondary_use',
        default='none'
    )

    # encodes categorical features
    encoder = BinaryEncoder()

    # scales numerical features
    scaler = RobustScaler()

    # trains and predicts on the transformed data
    estimator = DecisionTreeClassifier()
    # estimator = RandomForestClassifier()
    # estimator = LGBMClassifier()

    return [
        ('feature_remover', feature_remover),
        ('feature_engineering', feature_engineering),
        #('onehot_decoder_secondary_use', onehot_decoder_secondary_use),
        ('encoder_and_scaler', CustomColumnTransformer([
            ('encoder', encoder, make_column_selector(dtype_include=['category', 'object'])),
            ('scaler', scaler, make_column_selector(dtype_exclude=['category', 'object']))
        ], remainder='passthrough')),
        ('estimator', estimator)
    ]


class CustomPipeline:

    def __init__(
            self,
            steps,
            apply_ordinal_encoding=True,
            display_feature_importances=False,
            skip_evaluation=False
    ):
        self.steps = steps
        self.apply_ordinal_encoding = apply_ordinal_encoding
        self.display_feature_importances = display_feature_importances
        self.pipeline = Pipeline(steps=self.steps)
        self.skip_evaluation = skip_evaluation

    def run(self):
        print('loading data')
        #print(config.ROOT_DIR)
        
        X_train = pd.read_csv(os.path.join(config.ROOT_DIR, 'data/raw/train_values.csv'))
        y_train = pd.read_csv(os.path.join(config.ROOT_DIR, 'data/raw/train_labels.csv'))
        X_test = pd.read_csv(os.path.join(config.ROOT_DIR, 'data/raw/test_values.csv'))

        print('preparing data')

        X_train, y_train, X_test, X_test_building_id = self.clean(X_train, y_train, X_test)

        print('running pipeline')

        pipeline = self.pipeline #Pipeline(steps=self.steps)
        pipeline.fit(X_train, y_train)

        print('evaluating pipeline')
        if not(self.skip_evaluation):
            self.evaluate(pipeline, X_train, y_train)

        print('storing model and prediction')

        self.store(pipeline, X_test, X_test_building_id)

    def store(self, pipeline, X_test, X_test_building_id):
        # format prediction
        y_pred = pipeline.predict(X_test)
        if self.apply_ordinal_encoding:  # decode prediction if we applied ordinal/label encoding earlier
            y_pred = self.initial_label_encoder_.inverse_transform(y_pred)
        y_pred = pd.DataFrame({
            'building_id': X_test_building_id,
            'damage_grade': y_pred
        })

        # store trained model and prediction
        dump(pipeline, os.path.join(config.ROOT_DIR, 'models/tyrell_prediction.joblib'))
        y_pred.to_csv(os.path.join(config.ROOT_DIR, 'models/tyrell_prediction.csv'), index=False)

    def evaluate(self, pipeline, X_train, y_train):
        if self.display_feature_importances:
            X_train_preprocessed = pd.DataFrame(pipeline[:-1].transform(X_train))
            df = X_train_preprocessed.copy()
            df[y_train.name] = y_train
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(get_verbose_correlations(
                    df,
                    X_train_preprocessed.columns,
                    [y_train.name])
                )

            if isinstance(pipeline.named_steps['estimator'], LGBMClassifier):
                feature_names = pipeline.named_steps['estimator'].booster_.feature_name()
            else:
                feature_names = pipeline.named_steps['estimator'].feature_names_in_
            feature_importances = pipeline.named_steps['estimator'].feature_importances_
            df = pd.DataFrame(zip(feature_names, feature_importances), columns=['feature', 'importance'])
            df = df.sort_values(by=['importance'], ascending=False, ignore_index=True)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df)

        scoring = {
            'accuracy': 'accuracy',
            'f1-score': 'f1_macro',
            'mcc': make_scorer(matthews_corrcoef)
        }
        scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5)
        for score in scores:
            print('    ' + score + ':', scores[score].mean())

    def clean(self, X_train, y_train, X_test, recalculateOutliers = False):
        # store building_id of test set as it is required in the submission format of the prediction
        X_test_building_id = X_test['building_id']

        # remove building_id from target
        X_train = X_train.merge(y_train)
        y_train = X_train['damage_grade']
        X_train = X_train.drop(columns=['damage_grade'])

        categorical_columns = config.categorical_columns

        numerical_columns = config.numerical_columns
        
        has_secondary_use_columns = config.has_secondary_use_columns
    
        has_superstructure_columns = config.has_superstructure_columns

        if self.apply_ordinal_encoding:
            # apply an initial ordinal encoding on the categorical features
            self.initial_ordinal_encoder_ = OrdinalEncoder(cols=categorical_columns)
            X_train = self.initial_ordinal_encoder_.fit_transform(X_train)
            X_test = self.initial_ordinal_encoder_.transform(X_test)

            self.initial_label_encoder_ = LabelEncoder()
            y_train = pd.Series(
                data=self.initial_label_encoder_.fit_transform(y_train),
                index=y_train.index,
                name=y_train.name
            )

        # update data types
        X_train[categorical_columns] = X_train[categorical_columns].astype('category')
        X_train[numerical_columns] = X_train[numerical_columns].astype(np.float64)
        y_train = y_train.astype('category')
        X_test[categorical_columns] = X_test[categorical_columns].astype('category')
        X_test[numerical_columns] = X_test[numerical_columns].astype(np.float64)

        # --------- Outlier Removal -----------
        # rows we found to contain outliers which can therefore be dropped
        # remove the has_secondary_use and has_superstructure columns to not run analysis on them
        new_categorical_columns = list(set(categorical_columns) - set(has_superstructure_columns) - set(has_secondary_use_columns))
        
        row_indizes_to_remove = build_features.get_outlier_rows_as_index(X_train, numerical_columns, new_categorical_columns, 0.2)
        X_train = build_features.remove_rows_by_integer_index(X_train, row_indizes_to_remove)
        y_train = build_features.remove_rows_by_integer_index(y_train, row_indizes_to_remove)
        
        if len(X_train) != len(y_train):
            print('Error: X_train is not equal length to y_train!')

        # columns we found to be uninformative which can therefore be dropped
        columns_to_remove = [
            'building_id',
            'has_secondary_use',
            'plan_configuration',
            'has_superstructure_stone_flag',
            'has_superstructure_mud_mortar_brick',
            'has_superstructure_rc_non_engineered',
            'legal_ownership_status',
            'count_families'
            'has_superstructure_cement_mortar_stone',
            'has_superstructure_rc_engineered', 
            'has_superstructure_other'
        ]
        feature_remover = RemoveFeatureTransformer(features_to_drop=columns_to_remove)
        X_train = feature_remover.fit_transform(X_train, y_train)
        X_test = feature_remover.transform(X_test)

        # update categorical and numerical column lists
        # for column in columns_to_remove:
        #     if column in categorical_columns:
        #         categorical_columns.remove(column)
        #     if column in numerical_columns:
        #         numerical_columns.remove(column)

        return X_train, y_train, X_test, X_test_building_id
