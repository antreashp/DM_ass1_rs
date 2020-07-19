import pandas as pd



df = pd.read_csv("../data/dataset_mood_smartphone.csv")
print(df)

df1 = df[(df['value'] >= 0.1) & (df['variable'] == 'mood')]
print(df1)