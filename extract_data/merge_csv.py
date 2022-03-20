import os
import pandas as pd

DIRECTORY = "extracted_data"
FILES = [os.path.join(DIRECTORY, file) for file in os.listdir(DIRECTORY) if "csv" in file]
FILES.sort()

df = pd.read_csv(FILES.pop(0), index_col=0)

for file in FILES:
    df = pd.concat([df, pd.read_csv(file, index_col=0)], ignore_index=True)

df.to_csv("data.csv", index=False)
