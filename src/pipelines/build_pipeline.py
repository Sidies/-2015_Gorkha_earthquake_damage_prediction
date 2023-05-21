import numpy as np
import pandas as pd
import tqdm as tqdm
import matplotlib.pyplot as plt
import os

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from copy import deepcopy
from joblib import dump
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, make_scorer,\
    matthews_corrcoef
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from collections import Counter
from src.data import configuration as config
from src.features import handle_outliers
from src.features import build_features
from src.pipelines import pipeline_cleaning
from src.features import sampling_strategies

class CustomPipeline:
    """
    A custom pipeline class for running machine learning pipelines. 
    
    This class allows for various steps including data loading, preparation, cleaning, 
    evaluation, and storing of predictions. It also includes verbose output, 
    configuration of cleaning, evaluation and prediction steps, and the use of k-fold shuffling.
    """
    X_train = pd.DataFrame()
    y_train = pd.DataFrame()
    test_values = pd.DataFrame()
    X_train_raw = pd.DataFrame()
    y_train_raw = pd.DataFrame()
    test_values_raw = pd.DataFrame()
    test_values_building_id = []
    evaluation_scoring = {}
    pipeline_steps = []
    outlier_handler: pipeline_cleaning.OutlierHandler
    resampler: sampling_strategies.Sampler

    def __init__(
            self,
            force_cleaning=True,
            skip_storing_cleaning=False,
            skip_evaluation=False,
            skip_error_evaluation=True,
            skip_feature_evaluation=True,
            print_evaluation=True,
            skip_storing_prediction=False,
            use_kfold_shuffle=False,
            apply_coordinate_mapping=False,
            verbose=1
    ):
        """
        Initializes the CustomPipeline instance with specific steps and configurations.

        :param steps: List of (name, transform) tuples specifying the pipeline steps.
        :param force_cleaning: Whether to force data cleaning, defaults to False.
        :param skip_storing_cleaning: Whether to skip storing of cleaned data, defaults to False.
        :param skip_evaluation: Whether to skip evaluation of the pipeline, defaults to False.
        :param skip_error_evaluation: Whether to skip error evaluation, defaults to True.
        :param skip_feature_evaluation: Whether to skip feature evaluation, defaults to True.
        :param print_evaluation: Whether to print the evaluation, defaults to True.
        :param skip_storing_prediction: Whether to skip storing of prediction, defaults to False.
        :param use_kfold_shuffle: Whether to use k-fold shuffling in evaluation, defaults to False.
        :param apply_coordinate_mapping: Whether to apply coordinate mapping to geo_level_1_id
        :param verbose: Verbosity level of the output that describes how much should printed to terminal, defaults to 1.
        """
        self.force_cleaning = force_cleaning
        self.skip_storing_cleaning = skip_storing_cleaning
        self.skip_evaluation = skip_evaluation
        self.skip_error_evaluation = skip_error_evaluation
        self.skip_feature_evaluation = skip_feature_evaluation
        self.print_evaluation = print_evaluation
        self.skip_storing_prediction = skip_storing_prediction
        self.verbose = verbose
        self.use_kfold_shuffle = use_kfold_shuffle
        self.apply_coordinate_mapping = apply_coordinate_mapping
        
        # add default outlier handler
        # self.outlier_handler = pipeline_cleaning.OutlierRemover(cat_threshold=0.26, zscore_threshold=2.3)
        self.outlier_handler = pipeline_cleaning.OutlierHandler()
        
        # add default resampler
        self.resampler = sampling_strategies.Sampler()
        
        # load the data
        if self.verbose >= 1:
            print('loading data')
        self.load_and_prep_data()


    def add_new_step(self, transformer, name):
        """Adds a new step to the pipeline at the second to last position if the name is not already present.
        If the name is already present, the step with this name will be replaced.

        Args:
            transformer (transformer): the transformer to be added
            name (string): name of the step
        """
        # Check if there is an step already defined with this name
        for i, step in enumerate(self.pipeline_steps):
            if step[0] == name:
                # Replace the step
                self.pipeline_steps[i] = (name, transformer)
                return
        steps = self.pipeline_steps
        position = len(steps) - 1  
        steps.insert(position, (name, transformer))
        self.pipeline_steps = steps
    
    
    def remove_step(self, name):
        """Removes a specific step from the pipeline

        Args:
            name (string): the name of the step to remove
        """
        self.pipeline_steps.pop(name, None)
    
    
    def change_estimator(self, new_estimator):
        """Changes the current estimator of the pipeline to a different one. 
        If no estimator is defined a new one will be added.

        Args:
            new_estimator (model): the new model that should be applied
        """
        # Check if there is an 'estimator' step already defined
        for i, step in enumerate(self.pipeline_steps):
            if step[0] == 'estimator':
                # Replace the estimator
                self.pipeline_steps[i] = ('estimator', new_estimator)
                break
        else:
            # If no 'estimator' step was found, add one
            self.pipeline_steps.append(('estimator', new_estimator))

    def apply_outlier_handler(self, handler: pipeline_cleaning.OutlierHandler):
        """Changes the current outlier handler to a different one.

        Args:
            handler (pipeline_cleaning.OutlierHandler): The outlier handler that should be used in the cleaning step.
        """
        self.outlier_handler = handler
        
    
    def apply_sampler(self, resampler: sampling_strategies.Sampler):
        
        self.resampler = resampler
    
    
    def run(self):
        """
        Runs the entire pipeline including data loading, preparation, cleaning, 
        fitting the model, evaluation, and storing of prediction.
        """
        self.pipeline = ImbPipeline(self.pipeline_steps)
        
        # Split the data into train and test sets
        self.X_train, X_test, self.y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
        
        # prepare the data
        if self.verbose >= 1:
            print('preparing data')
        if self.force_cleaning:
            self.clean()

        if self.verbose >= 1:
            print('running pipeline')
        self.pipeline.fit(self.X_train, self.y_train)

        if not self.skip_evaluation:
            if self.verbose >= 1:
                print('evaluating pipeline')
            self.evaluate()

        if not self.skip_storing_prediction:
            if self.verbose >= 1:
                print('storing model and prediction')
            self.store()
            
        # ---------- Step 3: Evaluate on the test set ----------
        y_pred = self.pipeline.predict(X_test)
        test_mcc = matthews_corrcoef(y_test['damage_grade'], y_pred)
        print("Trained model MCC Score on unseen data:", test_mcc)


    def load_and_prep_data(self):
        """
        Loads and prepares the data. It reads the raw data, updates data types, and 
        checks if cleaning is required.
        """
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.test_values = pd.DataFrame()
        self.test_values_building_id = []
        # loading the raw uncleaned data
        self.X_train_raw = pd.read_csv(os.path.join(config.ROOT_DIR, 'data/raw/train_values.csv'))
        self.y_train_raw = pd.read_csv(os.path.join(config.ROOT_DIR, 'data/raw/train_labels.csv')).squeeze()
        self.test_values_raw = pd.read_csv(os.path.join(config.ROOT_DIR, 'data/raw/test_values.csv'))
        
        if not self.force_cleaning:
            # if the force cleaning flag is not set, the cleaned and prepared data from the interim folder is loaded
            test_values_path = Path(os.path.join(config.ROOT_DIR, 'data/interim/X_test.csv'))
            train_values_path = Path(os.path.join(config.ROOT_DIR, 'data/interim/X_train.csv'))
            target_values_path = Path(os.path.join(config.ROOT_DIR, 'data/interim/y_train.csv'))
            test_values_building_id_path = Path(os.path.join(config.ROOT_DIR, 'data/interim/X_test_building_id.csv'))

            # in the following it will be checked whether the paths contain files and the files are loaded
            # the data types will also be updated
            if test_values_path.is_file() and train_values_path.is_file() and target_values_path.is_file() and test_values_building_id_path.is_file():
                self.test_values = pd.read_csv(test_values_path)
                self.X_train = pd.read_csv(train_values_path)
                self.y_train = pd.read_csv(target_values_path).squeeze()
                self.test_values_building_id = pd.read_csv(test_values_building_id_path).squeeze("columns")

                # update data types
                categorical_columns = list(set(self.X_train.columns) - set(config.numerical_columns))
                numerical_columns = list(set(self.X_train.columns) - set(config.categorical_columns))
                self.X_train[categorical_columns] = self.X_train[categorical_columns].astype('category')
                self.X_train[numerical_columns] = self.X_train[numerical_columns].astype(np.float64)
                self.y_train = self.y_train.astype('category')
                self.test_values[categorical_columns] = self.test_values[categorical_columns].astype('category')
                self.test_values[numerical_columns] = self.test_values[numerical_columns].astype(np.float64)
        
        # if there is an issue with loading the prepared data or the data is not present the raw data will be used instead
        # the cleaning function is then triggered
        if len(self.X_train) <= 0 or len(self.y_train) <= 0 or len(self.test_values) <= 0 or len(
                self.test_values_building_id) <= 0 or self.force_cleaning:
            self.force_cleaning = True
            self.X_train = self.X_train_raw
            self.y_train = self.y_train_raw
            self.test_values = self.test_values_raw
            

    def clean(self):
        """
        Cleans the data by removing outliers, unnecessary features, and doing one-hot decoding. 
        It also stores the cleaned data if not instructed otherwise.
        """
        
        X_test = self.test_values
        X_train = self.X_train
        y_train = self.y_train

        # store building_id of test set as it is required in the submission format of the prediction
        X_test_building_id = X_test['building_id']

        # remove building_id from target
        X_train = pd.merge(X_train, y_train)
        y_train = X_train['damage_grade']
        X_train = X_train.drop(columns=['damage_grade'])

        categorical_columns = config.categorical_columns
        numerical_columns = config.numerical_columns

        # update data types
        X_train[categorical_columns] = X_train[categorical_columns].astype('category')
        X_train[numerical_columns] = X_train[numerical_columns].astype(np.float64)
        y_train = y_train.astype('category')
        X_test[categorical_columns] = X_test[categorical_columns].astype('category')
        X_test[numerical_columns] = X_test[numerical_columns].astype(np.float64)

        # --------- Outlier Removal -----------
        X_train, y_train = self.outlier_handler.handle_outliers(X_train=X_train, y_train=y_train)
        

        # --------- Feature Removal -----------

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

        feature_remover = build_features.RemoveFeatureTransformer(features_to_drop=columns_to_remove)
        X_train = feature_remover.fit_transform(X_train, y_train)
        X_test = feature_remover.transform(X_test)

        # --------- One-Hot Decoding -----------

        # decode onehot-encoded secondary use features
        onehot_decoder_secondary_use = build_features.OneHotDecoderTransformer(
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
            new_feature='has_secondary_use',
            default='none'
        )
        X_train = onehot_decoder_secondary_use.fit_transform(X_train, y_train)
        X_test = onehot_decoder_secondary_use.transform(X_test)

        # --------- Coordinate Mapping -----------

        # map geo-level-1-ids to their corresponding coordinates
        if self.apply_coordinate_mapping:
            geo_level_coordinate_mapper = build_features.GeoLevelCoordinateMapperTransformer()
            X_train = geo_level_coordinate_mapper.fit_transform(X_train, y_train)
            X_test = geo_level_coordinate_mapper.transform(X_test)
            
        # --------- Sampling ----------------------
        X_train, y_train = self.resampler.apply_sampling(X_train=X_train, y_train=y_train)

        # ---------- Store Cleaned Dataset ----------

        if not self.skip_storing_cleaning:
            print('storing cleaned data')
            X_train.to_csv(os.path.join(config.ROOT_DIR, 'data/interim/X_train.csv'), index=False)
            y_train.to_csv(os.path.join(config.ROOT_DIR, 'data/interim/y_train.csv'), index=False)
            X_test.to_csv(os.path.join(config.ROOT_DIR, 'data/interim/X_test.csv'), index=False)
            X_test_building_id.to_csv(os.path.join(config.ROOT_DIR, 'data/interim/X_test_building_id.csv'), index=False)

        self.X_train = X_train
        self.y_train = y_train
        self.test_values = X_test
        self.test_values_building_id = X_test_building_id
    

    def evaluate(self):
        """
        Evaluates the pipeline using performance metrics, error evaluation and feature evaluation. 
        It stores the scores and optionally prints the evaluation results.
        """
        scores = {}

        # use a copy to avoid overwriting the training of the original pipeline
        pipeline = deepcopy(self.pipeline)

        # ---------- Performance Metrics ----------

        performance_metrics = {
            'accuracy': 'accuracy',
            'f1-score': 'f1_macro',
            'mcc': make_scorer(matthews_corrcoef)
        }

        performance_scores = cross_validate(
            pipeline,
            self.X_train,
            self.y_train,
            scoring=performance_metrics,
            cv=StratifiedKFold(n_splits=5, shuffle=self.use_kfold_shuffle)
        )

        if self.print_evaluation:
            for metric in performance_scores:
                if self.verbose >= 1:
                    print('    ' + metric + ':', performance_scores[metric].mean())
                else:
                    print(metric + ':', performance_scores[metric].mean())

        scores.update(performance_scores)

        # ---------- Error Evaluation ----------
        if not self.skip_error_evaluation:
            X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, stratify=self.y_train)
            pipeline.fit(X_train, y_train)

            cm = confusion_matrix(y_test, pipeline.predict(X_test))
            scores['confusion_matrix'] = cm

            if self.print_evaluation:
                ConfusionMatrixDisplay(cm).plot()
                plt.show()

        if not self.skip_feature_evaluation:
            # get preprocessed train set
            df = pd.DataFrame(pipeline[:-1].transform(self.X_train)).copy()

            # store feature names
            feature_names = df.columns
            target_name = self.y_train.name

            # ---------- Feature Target Correlation ----------

            # # merge train set and target for easier analysis
            # df[target_name] = self.y_train
            #
            # scores['feature_target_correlations'] = get_verbose_correlations(
            #     df,
            #     feature_names,
            #     [target_name]
            # )
            #
            # if self.print_evaluation:
            #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #         print(scores['feature_target_correlations'])

            # ---------- Estimator Feature Importance ----------

            if hasattr(pipeline.named_steps['estimator'], 'feature_importances_'):
                # get importance of features for estimator
                feature_importances = pipeline.named_steps['estimator'].feature_importances_

                # merge importances with feature names
                df = pd.DataFrame(zip(feature_names, feature_importances), columns=['feature', 'importance'])
                df = df.sort_values(by=['importance'], ascending=False, ignore_index=True)

                scores['estimator_feature_importance'] = df

                if self.print_evaluation:
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        print(scores['estimator_feature_importance'].head(10))
                        print(scores['estimator_feature_importance'].tail(10))
            else:
                if self.print_evaluation:
                    print('no feature importances available')


        # safe to local class variable
        self.evaluation_scoring = scores

    def store(self):
        """
        Stores the trained model and the predictions, formatting the prediction into the required format.
        """
        # format prediction
        y_pred = self.pipeline.predict(self.test_values)
        y_pred = pd.DataFrame({
            'building_id': self.test_values_building_id,
            'damage_grade': y_pred
        })

        # store trained model and prediction
        dump(self.pipeline, os.path.join(config.ROOT_DIR, 'models/tyrell_prediction.joblib'))
        y_pred.to_csv(os.path.join(config.ROOT_DIR, 'models/tyrell_prediction.csv'), index=False)
