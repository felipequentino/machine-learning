import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import pydotplus
     
dataset = pd.read_csv('data/wine.data', header=None)

dataset.columns = ['label',
                   'alcohol',
                   'malic_acid',
                   'ash',
                   'alcalinity_of_ash',
                   'magnesium',
                   'total_phenols',
                   'flavanoids',
                   'nonflavanoid_phenols', 
                   'proanthocyanins', 
                   'color_intensity', 
                   'hue',
                   'OD280/OD315',
                   'proline']

print(dataset.head())

# Divisão treino-teste

from sklearn.model_selection import train_test_split

X = dataset.values[:, 1:]
y = dataset.values[:, 0] # a primeira coluna do dataset indica a origem do vinho
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train) # calcula a média e o desvio padrão para cada coluna

X_train = scaler.transform(X_train) # subtrai a média e divide pelo desvio padrão
X_test = scaler.transform(X_test) # subtrai a média e divide pelo desvio padrão

# Treinamento do modelo

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_model(height):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=height, random_state=0) #  cria o modelo
    model.fit(X_train, y_train) # treina o modelo
    return model

for height in range(1, 21): # testa diferentes alturas para a árvore
    model = train_model(height)
    y_pred = model.predict(X_test) # faz a predição

    print('--------------------------------------------------')
    print(f'Altura - {height}\n')
    print(confusion_matrix(y_test, y_pred)) # matriz de confusão
    print(f'Acurácia: {accuracy_score(y_test, y_pred)}') # acurácia

# Visualização da árvore de decisão

from IPython.display import Image
from sklearn.tree import export_graphviz

model = train_model(8) # treina o modelo com altura 3

feature_names = ['alcohol',
                 'malic_acid',
                 'ash',
                 'alcalinity_of_ash', 
                 'magnesium', 
                 'total_phenols', 
                 'flavanoids', 
                 'nonflavanoid_phenols', 
                 'proanthocyanins', 
                 'color_intensity', 
                 'hue',
                 'OD280/OD315',
                 'proline']

classes_names = ['%.f' % i for i in model.classes_]

dot_data = export_graphviz(model, filled=True, feature_names=feature_names, class_names=classes_names, rounded=True, special_characters=True) # cria o gráfico
graph = pydotplus.graph_from_dot_data(dot_data) # cria o gráfico

Image(graph.create_png()) # exibe o gráfico
graph.write_png('classification-tree.png') # salva o gráfico
Image('classification-tree.png')