import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
dataset = pd.read_csv('K means_clustering//Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values
from sklearn.cluster import KMeans  
#wcss_list= []
#for i in range(1, 11):  
#    model= KMeans(n_clusters=i, init='k-means++', random_state= 42)  
#    model.fit(x)  
#    wcss_list.append(model.inertia_)  
#plt.plot(range(1, 11), wcss_list)  
#plt.title('The Elobw Method Graph')
#plt.xlabel('Number of clusters(k)')  
#plt.ylabel('wcss_list')
#plt.savefig('K means_clustering//elbow_plot.png', bbox_inches='tight')
#plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)  
y_predict= kmeans.fit_predict(x)
plt.scatter(x[:,0], x[:,1], c=y_predict)
plt.title('Clusters of customers')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend()
plt.savefig('K means_clustering//cluster.png', bbox_inches='tight')
plt.show()