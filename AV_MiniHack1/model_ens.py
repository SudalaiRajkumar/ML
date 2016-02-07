import numpy as np
import pandas as pd

data_path = "./"
s1 = pd.read_csv(data_path + "sub_lr.csv")
s2 = pd.read_csv(data_path + "sub_xgb.csv")

s1["Count"] = 0.5*s1["Count"] + 0.5*s2["Count"]
s1.to_csv("sub_ens.csv", index=False)
