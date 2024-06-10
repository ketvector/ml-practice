import pandas as pd

simple_df = pd.DataFrame({
    "col1": ["v1", "v2"],
    "col2": [1, 2]
})

simple_df.info()


col1 = simple_df["col1"]

print(type(col1))