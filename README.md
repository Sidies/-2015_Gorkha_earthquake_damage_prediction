phase-1
==============================

Code, documentation and reports created for phase 1 of the data science lab 2023 at the Karlsruhe Institute of Technology.

Installation
------------

To run the following commands you need to have Python 3 and pip installed. 
After pulling the repository, open the terminal and run the following commands.

#### Linux

```
pip install -e .
pip install -r requirements.txt
```

#### Windows

```
py -m pip install -e .
py -m pip install -r requirements.txt
```

Running
-------

First, you need the data. 
You can either download it from the ilias lab course or from the <a target="_blank" href="https://www.drivendata.org/competitions/57/nepal-earthquake/">original source</a> directly.
Put the files in the `\data\raw` directory. 
The following files should be present:

- submission_format.csv
- test_values.csv
- train_labels.csv
- train_values.csv

Now, you are ready to run the prediction pipeline. 
To do this, enter the following command.

#### Linux

```
python3 main.py
```

#### Windows

```
py main.py
```

The resulting prediction file is `\models\tyrell_prediction.csv` and the trained model is stored as `\models\tyrell_prediction.joblib`.

### Arguments

`--feature_importance` 
Display additional information about feature importance.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   ├── pipelines      <- Scripts to build automated prediction pipelines
    │   │
    │   ├── tests          <- Scripts to validate the code with unit tests
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>