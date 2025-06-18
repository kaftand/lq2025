from Datasets import load_dataset
from experiment import run_trial
import numpy as np
import pandas as pd

experiments = []
for i_dataset in [0]:#range(3):
    for prev_base in [0.1]:#np.arange(0.1, 1.0, 0.1):
        trials = pd.DataFrame([run_trial(i_dataset, prev_base, i) for i in range(5000)])
        trials["i_dataset"] = i_dataset
        trials["prev_base"] = prev_base
        experiments.append(trials)

print(pd.concat(experiments).groupby(["i_dataset","prev_base"]).mean())
