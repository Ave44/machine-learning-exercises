import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# df = pd.read_csv("./Lab08/iris.csv") # dla Windows'a
df = pd.read_csv("iris.csv") # dla Linux'a

feature_cols = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
features = df[feature_cols]
labels = df['class']
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=13)


def runClassifier(n_neighbors):
    print("\n\n\n========================", n_neighbors, "========================\n")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(train_features, train_labels)

    print("Poprawność klayfikatora: ", knn.score(test_features, test_labels))

    # Macierz błędu
    class_true = []
    class_pred = knn.predict(test_features)

    for i in range(len(test_features)):
        class_true.append(test_labels.iloc[i])

    labels = ['setosa', 'versicolor', 'virginica']
    print('\nConfusion matrix:\n')
    matrix = confusion_matrix(class_true, class_pred, labels=labels)

    print("           | %-10s | %-10s | %-10s" % (labels[0], labels[1], labels[2]))
    print("----------------------------------------------------")
    for i in range(len(labels)):
        print("%-10s | %6d     | %6d     | %6d" % (labels[i], matrix[i][0], matrix[i][1], matrix[i][2]))
        print("----------------------------------------------------")


runClassifier(1)
runClassifier(3)
runClassifier(5)
runClassifier(7)
runClassifier(11)
