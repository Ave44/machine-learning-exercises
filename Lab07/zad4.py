import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Import danych
df = pd.read_csv("./Lab07/diabetes.csv") # dla Windows'a
# df = pd.read_csv("diabetes.csv") # dla Linux'a

feature_cols = ['pregnant-times','glucose-concentr','blood-pressure','skin-thickness','insulin','mass-index','pedigree-func','age']
features = df[feature_cols]
labels = df['class']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=1)
# print(train_features)

# Inicjalizacja i trenowanie drzewa
clf = tree.DecisionTreeClassifier()
clf.fit(train_features, train_labels)

# Sprawdzenie poprawności klasyfikatora
print("Poprawność klayfikatora: ", clf.score(test_features, test_labels))

# Wyświetlenie drzewa decyzyjnego w formie graficznej
tree.plot_tree(clf)
plt.show()

# Wyświetlenie macierzy błędów
class_true = []
class_pred = clf.predict(test_features)

for i in range(len(test_features)):
    class_true.append(test_labels.iloc[i])

labels = ['tested_positive','tested_negative']
print('\nConfusion matrix:\n')
matrix = confusion_matrix(class_true, class_pred, labels=labels)

print("                | %-15s | %-15s" % (labels[0], labels[1]))
print("----------------------------------------------------")
for i in range(len(labels)):
    print("%-15s | %8d        | %8d" % (labels[i], matrix[i][0], matrix[i][1]))
    print("----------------------------------------------------")