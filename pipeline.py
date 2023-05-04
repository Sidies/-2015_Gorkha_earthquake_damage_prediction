import numpy as np
import pandas as pd
import tqdm as tqdm

from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

from category_encoders.binary import BinaryEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder

from src.features.build_features import DummyTransformer, RemoveFeatureTransformer, DropRowsTransformer
from src.features import build_features

def run(extract_test_set=False, calledFromNotebook=False):
    goBackOneFolder = ''
    if calledFromNotebook:
        goBackOneFolder = '../'
    print('loading data')

    X_train = pd.read_csv(goBackOneFolder + 'data/raw/train_values.csv')
    y_train = pd.read_csv(goBackOneFolder + 'data/raw/train_labels.csv')
    if extract_test_set:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            stratify=y_train['damage_grade']
        )
    else:
        X_test = pd.read_csv(goBackOneFolder + 'data/raw/test_values.csv')

    # store building_id of test set as it is required in the submission format of the prediction
    X_test_building_id = X_test['building_id']

    print('preparing data')

    # remove building_id from target
    X_train = X_train.merge(y_train)
    y_train = X_train['damage_grade']
    X_train = X_train.drop(columns=['damage_grade'])
    if extract_test_set:
        X_test = X_test.merge(y_test)
        y_test = X_test['damage_grade']
        X_test = X_test.drop(columns=['damage_grade'])

    categorical_columns = [
        'building_id',
        'geo_level_1_id',
        'geo_level_2_id',
        'geo_level_3_id',
        'land_surface_condition',
        'foundation_type',
        'roof_type',
        'ground_floor_type',
        'other_floor_type',
        'position',
        'plan_configuration',
        'has_superstructure_adobe_mud',
        'has_superstructure_mud_mortar_stone',
        'has_superstructure_stone_flag',
        'has_superstructure_cement_mortar_stone',
        'has_superstructure_mud_mortar_brick',
        'has_superstructure_cement_mortar_brick',
        'has_superstructure_timber',
        'has_superstructure_bamboo',
        'has_superstructure_rc_non_engineered',
        'has_superstructure_rc_engineered',
        'has_superstructure_other',
        'legal_ownership_status',
        'has_secondary_use',
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
    ]

    numerical_columns = [
        'count_floors_pre_eq',
        'age',
        'area_percentage',
        'height_percentage',
        'count_families'
    ]

    # apply an initial ordinal encoding on the categorical features
    initial_ordinal_encoder = OrdinalEncoder(cols=categorical_columns)
    X_train = initial_ordinal_encoder.fit_transform(X_train)
    X_test = initial_ordinal_encoder.transform(X_test)

    initial_label_encoder = LabelEncoder()
    y_train = pd.Series(data=initial_label_encoder.fit_transform(y_train), index=y_train.index, name=y_train.name)
    if extract_test_set:
        y_test = pd.Series(data=initial_label_encoder.transform(y_test), index=y_test.index, name=y_test.name)

    # update data types
    X_train[categorical_columns] = X_train[categorical_columns].astype('category')
    X_train[numerical_columns] = X_train[numerical_columns].astype(np.float64)
    y_train = y_train.astype('category')
    X_test[categorical_columns] = X_test[categorical_columns].astype('category')
    X_test[numerical_columns] = X_test[numerical_columns].astype(np.float64)
    if extract_test_set:
        y_test = y_test.astype('category')

    # rows we found to contain outliers which can therefore be dropped
    rows_to_remove = [] #TODO

    # remove rows
    X_train = X_train.drop(index=rows_to_remove)
    y_train = y_train.drop(index=rows_to_remove)
        
    # columns we found to be uninformative which can therefore be dropped
    columns_to_remove = [
        'plan_configuration',
        'has_superstructure_stone_flag',
        'has_superstructure_mud_mortar_brick',
        'has_superstructure_rc_non_engineered',
        'legal_ownership_status',
        'count_families'
    ]

    # update categorical and numerical column lists
    for column in columns_to_remove:
        if column in categorical_columns:
            categorical_columns.remove(column)
        if column in numerical_columns:
            numerical_columns.remove(column)

    # ============================================= #
    # INITIALIZE SKLEARN PIPELINE TRANSFORMERS HERE #
    # ============================================= #

    print('running pipeline')

    # removes unnecessary columns
    feature_remover = RemoveFeatureTransformer(features_to_drop=columns_to_remove)

    # (maybe) impute missing values
    imputer = DummyTransformer()
    # imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    # (maybe) feature engineering
    feature_engineering = DummyTransformer()

    # (maybe) create artificial one-hot encoding by creating new features of columns where both have value 1, so we can
    # decode the one-hot-encoding in the next step
    one_hot_transformer = DummyTransformer()

    # (maybe) decode one-hot-encoded features
    one_hot_decoder = DummyTransformer()

    # encodes categorical features
    encoder = DummyTransformer()
    # encoder = OneHotEncoder(cols=categorical_columns)
    # encoder = BinaryEncoder(cols=categorical_columns)

    # scales numerical features
    scaler = DummyTransformer()
    # scaler = RobustScaler()

    # (maybe) data augmentation (balancing dataset regarding target)
    augmentation = DummyTransformer()

    # trains and predicts on the transformed data
    estimator = DummyClassifier()
    # estimator = DecisionTreeClassifier()
    # estimator = RandomForestClassifier()
    # estimator = LGBMClassifier()

    # define the pipeline
    pipeline = Pipeline(steps=[
        ('feature_remover', feature_remover),
        ('imputer', imputer),
        ('feature_engineering', feature_engineering),
        ('one_hot_transformer', one_hot_transformer),
        ('one_hot_decoder', one_hot_decoder),
        ('encoder_and_scaler', ColumnTransformer([
            ('encoder', encoder, categorical_columns),
            ('scaler', scaler, numerical_columns)
        ], remainder='passthrough')),
        ('augmentation', augmentation),
        ('estimator', estimator)
    ])

    
    # run the pipeline
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # only store prediction if real test set is used
    if extract_test_set:
        print('evaluate prediction')

        print('    accuracy:', accuracy_score(y_test, y_pred))
        print('    f1-score:', f1_score(y_test, y_pred, average='macro'))
        print('    mcc:', matthews_corrcoef(y_test, y_pred))
    else:
        print('storing models and predictions')

        # decode and format prediction
        y_pred = initial_label_encoder.inverse_transform(y_pred)
        y_pred = pd.DataFrame({
            'building_id': X_test_building_id,
            'damage_grade': y_pred
        })

        # store trained model and prediction
        dump(pipeline, goBackOneFolder + 'models/tyrell_prediction.joblib')
        y_pred.to_csv(goBackOneFolder + 'models/tyrell_prediction.csv', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run data cleaning, preprocessing and model training')
    parser.add_argument(
        '--extract_test_set',
        action='store_true',
        help='pass if you want to extract a test set from the train data to enable scoring'
    )
    parser.add_argument(
        '--ot', 
        type=float, 
        default=0.98, 
        help='Threshold value for the outlier detection')
    parser.add_argument(
        '--mi', 
        action='store_true', 
        help='If this flag is set to true, more information about the pipeline progress will be displayed')
    args = parser.parse_args()

    run(args.extract_test_set)
