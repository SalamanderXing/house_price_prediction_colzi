import os
import pandas as pd


data_path = os.path.join("data", "train.csv")
print(data_path)  # ./data/train.csv su linux e mac e .\data\train.csv su windows
data_frame = pd.read_csv(data_path)
