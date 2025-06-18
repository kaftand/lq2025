import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Dataset():
    name: str
    file: str
    target_col: str
    pos_value: any

datasets = [
    Dataset("diabetes", "diabetes.csv", "Outcome", 1),
    Dataset("breast_cancer","breast_cancer_data.csv","diagnosis", "M"),
    Dataset("HCV", "HCV-Egy-Data.csv", "Baselinehistological staging", 4)
]

def load_dataset(i_dataset):
    this_dataset = datasets[i_dataset]
    df = pd.read_csv(Path("./data")/ this_dataset.file)
    y = (df.loc[:,this_dataset.target_col] == this_dataset.pos_value).values
    df = df.dropna(axis=1)
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    X = df.drop(this_dataset.target_col, axis = 1).values
    return X, y

def simulate_dataset(seed, scale = 1, n_per_class = 1000):
    y = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)])
    X = np.concatenate([np.random.normal(loc=0, scale=scale, size=(n_per_class,1)),
                        np.random.normal(loc=0.85, scale=scale, size=(n_per_class,1))
                        ])
    return X, y