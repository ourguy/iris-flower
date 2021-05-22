#!/usr/bin/env python3
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train, y_train)


x_new = np.array([[5, 2.9, 1, 0.2]])
print("x_new.shape: {}".format(x_new.shape))

prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(dataset['target_names'][prediction]))
print("Test set score (knn.score): {:.2f}".format(knn.score(x_test, y_test)))
