import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def loadDataset(filename, split):
    dataset = pd.read_csv(filename)
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-split, random_state=0)
    return X_train, X_test, Y_train, Y_test


def euclideanDistance(array1, array2):
    dist = np.linalg.norm(array1 - array2)
    return dist


def main():
    X_train, X_test, Y_train, Y_test = loadDataset('iris.data', 0.67)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, Y_train)
    Y_pred = neigh.predict(X_test)
    score = neigh.score(X_test, Y_test)
    print('Score: %f' % score)

    accuracy = accuracy_score(Y_test, Y_pred)
    print('Accuracy: %f' % accuracy)


if __name__ == '__main__':
    main()
