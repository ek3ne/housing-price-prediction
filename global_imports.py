import os
import pandas as pd
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(_BASE_DIR, "Data", "train.csv")
test_path  = os.path.join(_BASE_DIR, "Data", "test.csv")

df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
