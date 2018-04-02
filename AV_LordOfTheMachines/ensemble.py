import pandas as pd
import numpy as np

s1 = pd.read_csv("../Submissions/srk_sub47.csv")
s2 = pd.read_csv("../Submissions/srk_sub48.csv")
#s3 = pd.read_csv("../Submissions/srk_sub23.csv")
#s4 = pd.read_csv("../Submissions/srk_sub24.csv")

#s1["is_click"] = 0.35*(0.5*s1["is_click"] + 0.5*s2["is_click"]) + 0.65*(0.65*(s3["is_click"])+0.35*(s4["is_click"]))
s1["is_click"] = 0.5*s1["is_click"] + 0.5*s2["is_click"]
s1.to_csv("srk_sub49.csv", index=False)
