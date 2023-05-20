from src.pipelines.build_pipeline import CustomPipeline
from src.pipelines import pipeline_utils


# disable warnings globally
import warnings
warnings.filterwarnings("ignore")


def run_best_performing_pipeline(
        force_cleaning=False,
        skip_storing_cleaning=False,
        skip_evaluation=False,
        skip_error_evaluation=True,
        skip_feature_evaluation=True,
        print_evaluation=True,
        skip_storing_prediction=False
):
    # create best performing pipeline
    pipeline = CustomPipeline(
        force_cleaning=force_cleaning,
        skip_storing_cleaning=skip_storing_cleaning,
        skip_evaluation=skip_evaluation,
        skip_error_evaluation=skip_error_evaluation,
        skip_feature_evaluation=skip_feature_evaluation,
        print_evaluation=print_evaluation,
        skip_storing_prediction=skip_storing_prediction
    )
    pipeline_utils.add_best_steps(pipeline)
    pipeline.run()
    
def run_lgbm_pipeline(
    force_cleaning=False,
    skip_storing_cleaning=False,
    skip_evaluation=False,
    skip_error_evaluation=True,
    skip_feature_evaluation=True,
    print_evaluation=True,
    skip_storing_prediction=False
):
    
    lgbm_pipeline = CustomPipeline(
        force_cleaning=force_cleaning,
        skip_storing_cleaning=skip_storing_cleaning,
        skip_evaluation=skip_evaluation,
        skip_error_evaluation=skip_error_evaluation,
        skip_feature_evaluation=skip_feature_evaluation,
        print_evaluation=print_evaluation,
        skip_storing_prediction=skip_storing_prediction
        )
    pipeline_utils.add_best_steps(custom_pipeline=lgbm_pipeline)
    pipeline_utils.add_lgbm_classifier(lgbm_pipeline)
    lgbm_pipeline.run()
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run data cleaning, preprocessing and model training')
    parser.add_argument(
        '--force-cleaning',
        action='store_true',
        help='pass if you want to force the cleaning step'
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
        "--pipeline", 
        choices=["best", "lgbm"], 
        default="best",
        help="Specify the pipeline to run"
    )
    args = parser.parse_args()
    
    if args.pipeline == "best":
        run_best_performing_pipeline(
            force_cleaning=args.force_cleaning,
            skip_evaluation=args.skip_evaluation,
            skip_error_evaluation=not args.error_evaluation,
            skip_feature_evaluation=not args.feature_importance,
            skip_storing_prediction=args.skip_storing_prediction
        )
    elif args.pipeline == "lgbm":
        run_lgbm_pipeline(
            force_cleaning=args.force_cleaning,
            skip_evaluation=args.skip_evaluation,
            skip_error_evaluation=not args.error_evaluation,
            skip_feature_evaluation=not args.feature_importance,
            skip_storing_prediction=args.skip_storing_prediction
        )
    else:
        print("Invalid pipeline specified.")

    
