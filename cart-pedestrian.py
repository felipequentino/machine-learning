import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Lendo o dataset
data = pd.read_csv('data/pedestrian.csv')
# Removendo as linhas com valores faltantes NA
data = data.dropna()

# REmovendo as duas primeiras colunas: Unnamed e Id
data = data.drop(['Unnamed: 0', 'id'], axis=1)
# Separando os dados em features e target
data_target = data['pedestrian_condition']
data_features = data.drop('pedestrian_condition', axis=1)

# Convertendo variáveis categóricas em dummies
data_features = pd.get_dummies(data_features)
# features names 
feature_names = data_features.columns.tolist()

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, test_size=0.3, random_state=1)

""" # Random Search para encontrar os melhores hiperparâmetros, utilizando RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Definindo os hiperparâmetros e as distribuições para a busca aleatória
param_dist = {"max_depth": [None] + list(np.arange(2, 20)),
              "min_samples_split": np.arange(2, 20),
              "min_samples_leaf": np.arange(1, 20),
              "criterion": ["gini", "entropy"]}

# Inicializando o classificador de árvore de decisão
tree = DecisionTreeClassifier()

# Inicializando a busca aleatória
random_search = RandomizedSearchCV(tree, 
                                   param_distributions=param_dist, n_iter=100, cv=3, random_state=0, n_jobs=-1)

# Executando a busca aleatória
random_search.fit(X_train, y_train)
# Imprimindo os melhores hiperparâmetros encontrados
print(random_search.best_params_) """

# Os melhores hiperparâmetros encontrados foram: {'min_samples_split': 14, 'min_samples_leaf': 16, 'max_depth': 11, 'criterion': 'entropy'}


# Scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train) # calcula a média e o desvio padrão para cada coluna
 
X_train = scaler.transform(X_train) # subtrai a média e divide pelo desvio padrão
X_test = scaler.transform(X_test) # subtrai a média e divide pelo desvio padrão

# Treinando o modelo CART
def train_model_cart(height):
    model = DecisionTreeClassifier(criterion='entropy',
    max_depth=height, min_samples_split=14, min_samples_leaf=16,random_state=0) #  cria o modelo
    model.fit(X_train, y_train) # treina o modelo
    return model

# Treinando o modelo M5
def train_model_m5():
    model = ExtraTreeClassifier(min_samples_leaf=10, min_samples_split=11) #  cria o modelo, na M5 a altura pode ser ilimitada
    model.fit(X_train, y_train) # treina o modelo
    return model

# Acurácia para diferentes alturas da árvore CART
print('---------------------- CART --------------------------')
for height in range(1, 20): # testa diferentes alturas para a árvore
    model = train_model_cart(height)
    y_pred = model.predict(X_test) # faz a predição

    print('--------------------------------------------------')
    print(f'Altura - {height}\n')
    print(confusion_matrix(y_test, y_pred)) # matriz de confusão
    print(f'Acurácia: {accuracy_score(y_test, y_pred)}') # acurácia

# A maior acurácia foi de 0.75, ALTURA 7

# Exibindo a árvore de decisão
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

model = train_model_cart(7) # treina o modelo com altura 11
classes_names = [str(i) for i in model.classes_]

dot_data = export_graphviz(model, filled=True, feature_names=feature_names, class_names=classes_names, rounded=True, special_characters=True) # cria o gráfico
graph = pydotplus.graph_from_dot_data(dot_data) 

Image(graph.create_png()) # exibe o gráfico
graph.write_png('cart-pedestrian.png') # salva o gráfico
Image('cart-pedestrian.png') 

# Agora realizar a análise desse dataset utilizando a árvore M5
# Acurácia para diferentes alturas da árvore M5
print('----------------------- M5 ---------------------------')
for epocs in range(1, 21):
    model = train_model_m5()
    y_pred = model.predict(X_test) # faz a predição

    print('--------------------------------------------------')
    print(f'Teste {epocs}\n')
    print(f'Acurácia: {accuracy_score(y_test, y_pred)}') # acurácia

# A acurácia em alguns casos (randômicos) a árvore M5 é melhor em comparação com a árvore CART
# A maior acurácia foi de 0.759, com 20 testes reproduzidos várias vezes.
model = train_model_m5() # treina o modelo com altura 11
classes_names = [str(i) for i in model.classes_]

dot_data = export_graphviz(model, filled=True, feature_names=feature_names, class_names=classes_names, rounded=True, special_characters=True) # cria o gráfico
graph = pydotplus.graph_from_dot_data(dot_data) 

Image(graph.create_png()) # exibe o gráfico
graph.write_png('m5-pedestrian.png') # salva o gráfico
Image('m5-pedestrian.png') 