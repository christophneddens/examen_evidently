"""monitoring of bike rental model with Evidently (exam).

Steps:
1. Download and extract the UCI Bike Sharing dataset.
2. Reference = January 2011, production = February 2011.
3. Train a RandomForestRegressor and validate it with a Report + tests.
4. Create the Workspace and the project.
5. One RegressionPreset snapshot per February week (W1, W2, W3).
"""
import os
import warnings
import zipfile

import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.core.datasets import Dataset, DataDefinition, Regression
from evidently.metrics import RMSE, R2Score
from evidently.presets import RegressionPreset
from evidently.tests import gt, lt
from evidently.ui.workspace import Workspace

# Hide library warnings so that the output stays readable
warnings.filterwarnings("ignore")

#settings
DATA_URL = ("https://archive.ics.uci.edu/static/public/275/"
            "bike+sharing+dataset.zip")
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "bike_sharing.zip")
CSV_PATH = os.path.join(DATA_DIR, "hour.csv")

WORKSPACE_PATH = "bike_sharing_monitoring"
PROJECT_NAME = "Bike Rental Monitoring"

TARGET = "cnt"
PREDICTION = "prediction"
NUM_FEATURES = ["temp", "atemp", "hum", "windspeed", "hr", "weekday"]
CAT_FEATURES = ["season", "holiday", "workingday", "weathersit"]
FEATURES = NUM_FEATURES + CAT_FEATURES

WEEKS = {
    "W1": ("2011-02-01", "2011-02-07"),
    "W2": ("2011-02-08", "2011-02-14"),
    "W3": ("2011-02-15", "2011-02-21"),
}


#1.ingestion
def download_data():
    """Download the zip file and extract it into data/ (only once)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        print("Downloading the dataset ...")
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)
    print(f"Data ready: {CSV_PATH}")


#2.Preparation
def select_dates(df, start, end):
    """Return the rows between two dates (both included)."""
    mask = (df["dteday"] >= start) & (df["dteday"] <= end)
    return df[mask].copy()


def load_data():
    """Return two DataFrames: January 2011 and February 2011."""
    df = pd.read_csv(CSV_PATH, parse_dates=["dteday"])
    reference = select_dates(df, "2011-01-01", "2011-01-31")
    production = select_dates(df, "2011-02-01", "2011-02-28")
    print(f"January rows: {len(reference)}, "
          f"February rows: {len(production)}")
    return reference, production


def make_dataset(df):
    """Wrap a DataFrame for Evidently: say which column is what."""
    definition = DataDefinition(
        numerical_columns=NUM_FEATURES + [TARGET, PREDICTION],
        categorical_columns=CAT_FEATURES,
        regression=[Regression(target=TARGET, prediction=PREDICTION)],
    )
    return Dataset.from_pandas(df, data_definition=definition)


#3.Training and validation
def train_model(january):
    """Split January 70/30, train the model, return model and test set."""
    train, test = train_test_split(january, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train[FEATURES], train[TARGET])
    test = test.copy()
    test[PREDICTION] = model.predict(test[FEATURES])
    return model, test


def validate_model(test):
    """Run a Report with tests on the January test set."""
    report = Report(
        metrics=[RMSE(tests=[lt(50)]), R2Score(tests=[gt(0.7)])],
        include_tests=True,
    )
    snapshot = report.run(current_data=make_dataset(test),
                          tags=["validation"])
    result = snapshot.dict()
    for test_result in result["tests"]:
        print(" ", test_result["status"], "-", test_result["description"])
    failed = [t for t in result["tests"] if t["status"] != "SUCCESS"]
    if failed:
        print("Model NOT validated for production.")
    else:
        print("Model validated for production.")
    return snapshot


#4. Workspace
def create_project():
    """Create the Workspace and a fresh project (an old one is deleted)."""
    ws = Workspace.create(WORKSPACE_PATH)
    for old_project in ws.search_project(PROJECT_NAME):
        ws.delete_project(old_project.id)
    project = ws.create_project(PROJECT_NAME)
    project.description = "January 2011 = reference, February = production"
    project.save()
    return ws, project


#5.Production simulation
def monitor_weeks(ws, project, model, reference, february):
    """Save one RegressionPreset snapshot per week; return data and RMSE."""
    weeks = {}
    for name, (start, end) in WEEKS.items():
        week = select_dates(february, start, end)
        week[PREDICTION] = model.predict(week[FEATURES])
        report = Report(metrics=[RegressionPreset()], include_tests=True)
        snapshot = report.run(
            current_data=make_dataset(week),
            reference_data=make_dataset(reference),
            timestamp=pd.Timestamp(start).to_pydatetime(),
            tags=[name],
        )
        ws.add_run(project.id, snapshot)
        rmse = root_mean_squared_error(week[TARGET], week[PREDICTION])
        weeks[name] = (week, rmse)
        print(f"{name} ({start} to {end}): {len(week)} rows, "
              f"RMSE = {rmse:.1f}")
    return weeks


def main():
    """Run all steps in order."""
    download_data()
    january, february = load_data()
    model, january_test = train_model(january)
    validation = validate_model(january_test)
    ws, project = create_project()
    ws.add_run(project.id, validation)
    weeks = monitor_weeks(ws, project, model, january_test, february)


if __name__ == "__main__":
    main()
