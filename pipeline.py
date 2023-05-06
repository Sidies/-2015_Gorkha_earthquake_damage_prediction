from src.pipelines.build_pipelines import CustomPipeline, get_best_steps

# disable warnings globally
import warnings
warnings.filterwarnings("ignore")


def run(display_feature_importance=False):
    pipeline = CustomPipeline(
        steps=get_best_steps(),
        apply_ordinal_encoding=False,
        display_feature_importances=display_feature_importance
    )
    pipeline.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run data cleaning, preprocessing and model training')
    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help='pass if you want to display feature importance information'
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

    run(display_feature_importance=args.feature_importance)
