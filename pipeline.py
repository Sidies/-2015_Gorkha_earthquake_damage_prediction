from src.pipelines.build_pipelines import CustomPipeline, get_best_steps

# disable warnings globally
import warnings
warnings.filterwarnings("ignore")


def run(force_cleaning=False, display_feature_importance=False):
    pipeline = CustomPipeline(
        steps=get_best_steps(),
        display_feature_importances=display_feature_importance,
        force_data_cleaning=force_cleaning
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
        '--force-cleaning',
        action='store_true',
        help='pass if you want to force the cleaning step'
    )
    args = parser.parse_args()

    run(
        force_cleaning=args.force_cleaning,
        display_feature_importance=args.feature_importance
    )
