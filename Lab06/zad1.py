import pandas as pd
import numpy as np

missing_values = ["n/a", "na", "--", "-"]
# df = pd.read_csv("./Lab06/iris_with_errors.csv", na_values = missing_values) # dla Windows'a
df = pd.read_csv("iris_with_errors.csv", na_values = missing_values) # dla Linux'a

# a i b

print("=== Po wczytaniu danych ===")
print(df.isnull().sum())
print("Total amount of mistakes: ", df.isnull().sum().sum())

# funkcja naprawia wszystkie puste pola i dane wychodzące poza skalę (0,15)
def fillMissingValues(column):
    # zamiana wszystkich niezgodnuch pól na Nan
    for i in range(len(df[column])):
        try:
            float(df[column][i])
            if df[column][i] >= 15 or df[column][i] <= 0:
                df.loc[i, column] = np.nan
        except ValueError:
            df.loc[i, column] = np.nan
            pass

    # zamiana wsztstki pól Nan na medianę  
    median = df[column].median()
    for i in range(len(df[column])):
        if np.isnan(df[column][i]):
            df.loc[i, column] = median

fillMissingValues('sepal.length')
fillMissingValues('sepal.width')
fillMissingValues('petal.length')
fillMissingValues('petal.width')

print("\n=== Po uzupełnieniu danych ===\n", df.isnull().sum(), sep="")
print("Total amount of mistakes: ", df.isnull().sum().sum())
print("Czy zostały jakieś nieuzupełnione dane?: ", df.isnull().values.any())

# c

print("\n=== Wszystkie błędne nazwy ===")
correctNames = ['Setosa', 'Versicolor', 'Virginica']
incorrectNames = []
for i in df['variety']:
    if (not i in incorrectNames) and (not i in correctNames):
        incorrectNames.append(i)
print(incorrectNames)

# zamiana wszystkich błędnych nazw na ich poprawne odpowiedniki
correcting = {'setosa': 'Setosa', 'Versicolour': 'Versicolor', 'VersiColor': 'Versicolor', 'virginica': 'Virginica'}
for i in range(len(df['variety'])):
    if df['variety'][i] in incorrectNames:
        df.loc[i, 'variety'] = correcting[df['variety'][i]]

print("\n=== Nazwy które zostały po oczyszczeniu ===")
allNames = []
for i in df['variety']:
    if not i in allNames:
        allNames.append(i)
print(allNames)
