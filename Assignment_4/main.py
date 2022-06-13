import itertools
import numpy as np
import pandas as pd
from paper_code.paper_wrapper import PaperWrapper

#  LOAD DATA
inner_func = [(f"F{idx+1}R", f"F{idx+1}S") for idx in range(22)]
columns = ["OVERALL_DIAGNOSIS"] + list(itertools.chain(*inner_func))
with open("./Data/SPECTF.train") as f:

    lines = f.readlines()
    data_array = np.zeros((len(lines), 45))
    for row_idx, line in enumerate(lines):
        data_array[row_idx] = list(
            map(lambda x: int(x), line.replace("\n", "").split(","))
        )

    data_dict = {col_name: data for col_name, data in zip(columns, data_array)}
    data = pd.DataFrame.from_dict(data_dict)
    data.head()
    solver = PaperWrapper()
    X, y = data.iloc[:, 1:], data.iloc[:, 0]
    print(solver(X, y))
