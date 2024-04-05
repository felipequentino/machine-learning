import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from Kmean import Kmeans

sns.set_context('notebook')
plt.style.use('fivethirtyeight')
from warnings import filterwarnings
filterwarnings('ignore')

# Carregando os dados
df = pd.read_csv('data/clientes_shopping.csv')

# Dropado as colunas que não serão utilizadas, já que o par do Cluster é Renda e Escore
arr_drop = ['ID', 'Genero', 'Idade']
df.drop(arr_drop, axis=1, inplace=True)

# Padronizando os dados
X_std = StandardScaler().fit_transform(df)

# Usando o método do cotovelo para encontrar o número ideal de clusters
# O cotovelo é o ponto onde a linha começa a se aplanar, neste caso, o número ideal de clusters é 5
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_std)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 20), wcss)
plt.title('Método Cotovelo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.show()


# Usando o método da silhueta, que resultou em 5 clusters
from sklearn.metrics import silhouette_score

silhueta = [] 

for n_cluster in range(2, 13):
    silhueta.append( 
        silhouette_score(X_std, KMeans(n_clusters = n_cluster).fit_predict(X_std))) 
    
# Plotando um gráfico para visualizar os resultados
k = range(2,13)
plt.bar(k, silhueta) 
plt.xlabel('Número de clusters', fontsize = 10) 
plt.ylabel('Pontuação da Silhueta', fontsize = 10) 
plt.show() 

# Usando o método do gap estatístico, que resultou em 8-10 clusters
from optimalK import optimalK

k, gapdf = optimalK(X_std, 15)
print(f'Número ideal de clusters: {k}')


# Rodando a implementação local do kmeans
km = Kmeans(n_clusters=5, max_iter=100)

km.fit(X_std)
centroids = km.centroids

# Plotando os dados clusterizados
fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,
            c='r', label='centroide')
plt.scatter(X_std[km.labels == 0, 0], X_std[km.labels == 0, 1],
            c='green', label='cluster 1')
plt.scatter(X_std[km.labels == 1, 0], X_std[km.labels == 1, 1],
            c='blue', label='cluster 2')
plt.scatter(X_std[km.labels == 2, 0], X_std[km.labels == 2, 1],
            c='orange', label='cluster 3')
plt.scatter(X_std[km.labels == 3, 0], X_std[km.labels == 3, 1],
            c='purple', label='cluster 4')
plt.scatter(X_std[km.labels == 4, 0], X_std[km.labels == 4, 1],
            c='yellow', label='cluster 5')


plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('Renda')
plt.ylabel('Escore')
plt.title('Visualização dos clusters', fontweight='bold')
ax.set_aspect('equal')
plt.show()