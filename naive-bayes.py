# Tal qual o knn, será usado o dataset iris para treinar o modelo, por conta
# do alvo ser uma variável categórica. Mas agora, utilizando Naive-Bayes.

import random
random.seed(42) # define a seed para reproduzir os resultados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('machine-learning/iris/iris.data', header=(0))
data = data.dropna(axis='rows') # remove linhas com valores nulos ( NaN )

# armazena os nomes das classes
classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)
print("Número de linhas e colunas na matriz d atributos: ", data.shape)
attributes = list(data.columns)

data = data.to_numpy() # transforma os dados em um array numpy
nrow, ncol = data.shape
y = data[:,-1] # vetor de classes/rótulos, é o vetor destino, o nosso alvo
X = data[:,0:ncol-1] # matriz de atributos/características

# Selecionando os conjuntos de treinmaneto e teste
from sklearn.model_selection import train_test_split
p = 0.7 # fração de elementos no conjunto de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=p, random_state=42)

# definição de uma função para calcular a densidade de probabilidade conjunta 
# definição de função de verossimilhança

def likelyhood(y, Z):
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    prob = 1
    for j in np.arange(0, Z.shape[1]):
        m = np.mean(Z[:,j])
        s = np.std(Z[:,j])
        prob = prob*gaussian(y[j], m, s)
    return prob

# cálculo da estimação para cada classe:
P = pd.DataFrame(data=np.zeros((X_train.shape[0], len(classes))), columns=classes)
for i in np.arange(0, len(classes)):
    elements = tuple(np.where(y_train == classes[i]))
    Z = X_train[elements,:][0]
    for j in np.arange(0, X_test.shape[0]):
        x = X_test[j,:]
        pj = likelyhood(x, Z)
        P[classes[i]][j] = pj*len(elements)/X_train.shape[0]

print(P.head(10))

# calcula a acurácia
from sklearn.metrics import accuracy_score

y_pred = []
for i in np.arange(0, X_test.shape[0]):
    y_pred.append(P.columns[np.argmax(P.iloc[i,:])])

y_pred = np.array(y_pred, dtype=str)
print("Acurácia: ", accuracy_score(y_test, y_pred)) 

# Matriz de confusão
# [[TP, FN 
#   FP, TN]]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusão: ", accuracy_score(y_test, y_pred))
print(cm)

# Usando o Naive-Bayes do sklearn

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_pred, y_test)
print("Acurácia: ", score)

df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})
print(df)
# Outra maneira de efetuarmos a classificação é assumirmos que
# os atributos possuem distribuição diferente do normal.
# UMa possibilidade é assumirmos que os dados possuem distribuição de Benroulli.

from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_pred, y_test)
print("Acurácia: ", score)

# Acurácia apenas de 0.28