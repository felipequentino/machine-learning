from sklearn import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Dataset iris
iris = load_iris()

# Alvo e features
X, y = iris.data, iris.target

# Classificador
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Plot 
plt.figure(figsize=(12, 12))
tree.plot_tree(clf)
plt.show()