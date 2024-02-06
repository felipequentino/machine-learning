from sklearn import datasets

import pandas as pd

iris = datasets.load_iris()

#print(iris.target_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target 

df['target_name'] = iris.target_names[df['target']]

iris_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Determinar nosso alvo (y), por ser uma classificação usando KNN,
# precisamos de um alvo númerico.

y = df.target         # alvo
X = df[iris_features] # features
# print(y)
# print(X)

# Usando o Scikit-learn:
# Define = Escolha de modelo (define parâmetros)
# Fit = Treinar
# Predict = Fazer a Predição
# Evaluate = avaliar os resultados

from sklearn.neighbors import KNeighborsClassifier

modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X, y)

print(modelo.predict(X))
print(modelo.score(X, y)) # acertou 144 e errou 6