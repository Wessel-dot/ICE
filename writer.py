import pandas as pd
import numpy as np
import os

def write(f_name, name, data):
    if os.path.isfile(f_name):
        df = pd.read_csv(f_name, index_col=0)
        new_data = pd.DataFrame(data, columns=range(data.shape[1]))
        new_data["name"] = name
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = pd.DataFrame(data, columns=range(data.shape[1]))
        df["name"] = name
    df.to_csv(f_name)

f_name = "Nic_Cage_faces.csv"
name = "Nic Cage"
data = np.random.rand(5,100)