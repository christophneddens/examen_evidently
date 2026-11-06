"""monitoring of bike rental model wit Evidently (exam).

Steps:
1. Download and extract the UCI Bike Sharing dataset.
2. Reference = January 2011, production = February 2011.
"""
import os
import warnings
import zipfile

import pandas as pd
import requests

#hide library warnings so that the output stays readable
warnings.filterwarnings("ignore")

#settings
DATA_URL = ("https://archive.ics.uci.edu/static/public/275/"
            "bike+sharing+dataset.zip")
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "bike_sharing.zip")
CSV_PATH = os.path.join(DATA_DIR, "hour.csv")


#1.Ingestion
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


#2preparation
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


def main():
    """Run all steps in order."""
    download_data()
    january, february = load_data()


if __name__ == "__main__":
    main()
