from src.pipelines.build_pipeline import CustomPipeline
from src.pipelines import pipeline_utils, pipeline_cleaning
from src.features import sampling_strategies


# disable warnings globally
import warnings
warnings.filterwarnings("ignore")


def run_best_performing_pipeline(
        force_cleaning=True,
        skip_storing_cleaning=False,
        skip_evaluation=False,
        skip_error_evaluation=True,
        skip_feature_evaluation=True,
        print_evaluation=True,
        skip_storing_prediction=False,
        use_validation_set=False,
        use_cross_validation=True,
        apply_coordinate_mapping=True,
        use_tuning=True
):
    """
    Initializes and starts a sklearn pipeline with the steps from the 
    best_performing_steps() method

    Args:
        force_cleaning (bool, optional): Whether to force the cleaning step or the presaved files. Defaults to False.
        skip_storing_cleaning (bool, optional): Whether to skop storing the cleaning data. Defaults to False.
        skip_evaluation (bool, optional): Whether the evaluation step should be performed. Defaults to False.
        skip_error_evaluation (bool, optional): Whether the error evaluation step should be performed. Defaults to True.
        skip_feature_evaluation (bool, optional): Whether the feature evaluation step should be performed. Defaults to True.
        print_evaluation (bool, optional): Whether the evaluation results should be printed. Defaults to True.
        skip_storing_prediction (bool, optional): Whether the prediction results should be printed. Defaults to False.
        use_validation_set (bool, optional): Whether to split the data into training and validation set for evaluation. Defaults to False.
    """
    pipeline = CustomPipeline(
        force_cleaning=force_cleaning,
        skip_storing_cleaning=skip_storing_cleaning,
        skip_evaluation=skip_evaluation,
        skip_error_evaluation=skip_error_evaluation,
        skip_feature_evaluation=skip_feature_evaluation,
        print_evaluation=print_evaluation,
        skip_storing_prediction=skip_storing_prediction,
        use_validation_set=use_validation_set,
        use_cross_validation=use_cross_validation,
        apply_coordinate_mapping=apply_coordinate_mapping,
        use_tuning=use_tuning
    )
    pipeline_utils.add_best_steps(pipeline)
    pipeline.run()
    
def run_lgbm_pipeline(
    force_cleaning=True,
    skip_storing_cleaning=False,
    skip_evaluation=False,
    skip_error_evaluation=True,
    skip_feature_evaluation=True,
    print_evaluation=True,
    skip_storing_prediction=False,
    use_validation_set=False,
    use_cross_validation=True,
    apply_coordinate_mapping=True,
    use_tuning=True
):
    """
    Initializes and starts a pipeline with the lgbm classifier
    """
    lgbm_pipeline = CustomPipeline(
        force_cleaning=force_cleaning,
        skip_storing_cleaning=skip_storing_cleaning,
        skip_evaluation=skip_evaluation,
        skip_error_evaluation=skip_error_evaluation,
        skip_feature_evaluation=skip_feature_evaluation,
        print_evaluation=print_evaluation,
        skip_storing_prediction=skip_storing_prediction,
        use_validation_set=use_validation_set,
        use_cross_validation=use_cross_validation,
        apply_coordinate_mapping=apply_coordinate_mapping,
        use_tuning=use_tuning
    )
    pipeline_utils.add_best_steps(custom_pipeline=lgbm_pipeline)
    pipeline_utils.apply_lgbm_classifier(lgbm_pipeline)
    lgbm_pipeline.run()
    
def run_multiple_pipelines(
    force_cleaning=True,
    skip_storing_cleaning=False,
    skip_evaluation=False,
    skip_error_evaluation=True,
    skip_feature_evaluation=True,
    print_evaluation=True,
    skip_storing_prediction=False,
    use_validation_set=False,
    use_cross_validation=True,
    apply_coordinate_mapping=True,
    use_tuning=True
):
    """
    Initializes and starts multiple pipelines with different classifiers
    """
    pipeline = CustomPipeline(
        force_cleaning=force_cleaning,
        skip_storing_cleaning=skip_storing_cleaning,
        skip_evaluation=skip_evaluation,
        skip_error_evaluation=skip_error_evaluation,
        skip_feature_evaluation=skip_feature_evaluation,
        print_evaluation=print_evaluation,
        skip_storing_prediction=skip_storing_prediction,
        use_validation_set=use_validation_set,
        use_cross_validation=use_cross_validation,
        apply_coordinate_mapping=apply_coordinate_mapping,
        use_tuning=use_tuning
    )
    n = 3
    
    print(f"Starting Pipeline (1/{n}) best steps: \n")
    pipeline_utils.add_best_steps(custom_pipeline=pipeline)
    pipeline.run()
    
    print(f"\n Starting Pipeline (2/{n}) lgbm: \n")
    pipeline_utils.apply_lgbm_classifier(pipeline)
    pipeline.run()
    
    print(f"\n Starting Pipeline (3/{n}) randomforest: \n")
    pipeline_utils.apply_randomforest_classifier(pipeline)
    pipeline.run()
    
def run_test_pipeline(
    force_cleaning=True,
    skip_storing_cleaning=False,
    skip_evaluation=False,
    skip_error_evaluation=True,
    skip_feature_evaluation=True,
    print_evaluation=True,
    skip_storing_prediction=False,
    use_validation_set=False,
    use_cross_validation=True,
    apply_coordinate_mapping=True,
    use_tuning=True
):
    """
    Initializes and starts a pipeline with the lgbm classifier
    """
    test_pipeline = CustomPipeline(
        force_cleaning=force_cleaning,
        skip_storing_cleaning=skip_storing_cleaning,
        skip_evaluation=skip_evaluation,
        skip_error_evaluation=skip_error_evaluation,
        skip_feature_evaluation=skip_feature_evaluation,
        print_evaluation=print_evaluation,
        skip_storing_prediction=skip_storing_prediction,
        use_validation_set=use_validation_set,
        apply_coordinate_mapping=apply_coordinate_mapping,
        use_cross_validation=use_cross_validation,
        use_tuning=use_tuning
    )

    pipeline_utils.add_outlier_handling(
        custom_pipeline=test_pipeline,
        outlier_handling_func=pipeline_cleaning.OutlierRemover(cat_threshold=0.26, zscore_threshold=2.3).handle_outliers
    )

    pipeline_utils.add_resampling(
        custom_pipeline=test_pipeline,
        resampling_func=sampling_strategies.RandomSampler().apply_sampling
    )

    pipeline_utils.add_binary_encoder_and_minmaxscaler(custom_pipeline=test_pipeline)

    # try out random sampling and see how it performs on the pipeline
    #pipeline_utils.add_randomsampling(custom_pipeline=test_pipeline)
    
    pipeline_utils.apply_lgbm_classifier(test_pipeline)
    test_pipeline.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run data cleaning, preprocessing and model training')
    parser.add_argument(
        '--no-force-cleaning',
        action='store_true',
        help='pass if you want to stop forcing the cleaning step'
    )
    parser.add_argument(
        '--validation-set',
        action='store_true',
        help='pass if you want to split the data into train and validation set'
    )
    parser.add_argument(
        '--no-cross-validation',
        action='store_true',
        help='pass if you want to skip cross validation in the evaluation'
    )
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='pass if you want to skip model evaluation'
    )
    parser.add_argument(
        '--error-evaluation',
        action='store_true',
        help='pass if you want to evaluate the errors'
    )
    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help='pass if you want to display feature importance'
    )
    parser.add_argument(
        '--skip-storing-prediction',
        action='store_true',
        help='pass if you want to skip storing the prediction'
    )
    parser.add_argument(
        '--no-coordinate-mapping',
        action='store_true',
        help='pass if you want to skip the geo data coordinate mapping'
    )
    parser.add_argument(
        '--tuning',
        action='store_true',
        help='pass if you want to skip hyperparameter tuning'
    )
    parser.add_argument(
        "--pipeline", 
        choices=["best", "lgbm", "test", "multiple"], 
        default="best",
        help="Specify the pipeline to run"
    )
    args = parser.parse_args()
    
    if args.pipeline == "best":
        run_best_performing_pipeline(
            force_cleaning=not args.no_force_cleaning,
            skip_evaluation=args.skip_evaluation,
            skip_error_evaluation=not args.error_evaluation,
            skip_feature_evaluation=not args.feature_importance,
            skip_storing_prediction=args.skip_storing_prediction,
            use_validation_set=args.validation_set,
            use_cross_validation=not args.no_cross_validation,
            apply_coordinate_mapping=not args.no_coordinate_mapping,
            use_tuning=args.tuning
        )
    elif args.pipeline == "lgbm":
        run_lgbm_pipeline(
            force_cleaning=not args.no_force_cleaning,
            skip_evaluation=args.skip_evaluation,
            skip_error_evaluation=not args.error_evaluation,
            skip_feature_evaluation=not args.feature_importance,
            skip_storing_prediction=args.skip_storing_prediction,
            use_validation_set=args.validation_set,
            use_cross_validation=not args.no_cross_validation,
            apply_coordinate_mapping=not args.no_coordinate_mapping,
            use_tuning=args.tuning
        )
    elif args.pipeline == "test":
        run_test_pipeline()
    elif args.pipeline == "multiple":
        run_multiple_pipelines(
            force_cleaning=not args.no_force_cleaning,
            skip_evaluation=args.skip_evaluation,
            skip_error_evaluation=not args.error_evaluation,
            skip_feature_evaluation=not args.feature_importance,
            skip_storing_prediction=args.skip_storing_prediction,
            use_validation_set=args.validation_set,
            use_cross_validation=not args.no_cross_validation,
            apply_coordinate_mapping=not args.no_coordinate_mapping,
            use_tuning=args.tuning
        )
    else:
        print("Invalid pipeline specified.")

    
