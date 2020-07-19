import pandas
import os

data = pandas.read_csv('data/dataset_mood_smartphone.csv', index_col=0)

variables = data['variable'].unique()
dataframes = {

}

for variable in variables:
    variable_dataframe = data[data['variable'] == variable].drop('variable', axis=1)
    users = variable_dataframe['id'].unique()

    dataframes[variable] = {}
    for user in users:
        dataframes[variable][user] = data[
            (data['variable'] == variable) & (data['id'] == user)
        ].drop('id', axis=1).drop('variable', axis=1)

for variable in dataframes.keys():
    for user in dataframes[variable].keys():
        if not os.path.exists('data/' + user.replace('.', '_')):
            os.mkdir('data/' + user.replace('.', '_'))
        dataframes[variable][user].to_csv('data/' + user.replace('.', '_') + '/' + variable.replace('.', '_') + '.csv')


# print(dataframes)
print(dataframes['mood']['AS14.01'])
# print(raw_data)
