import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
dataset = pd.read_csv('Hierarchical_clustering//Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values
from sklearn.cluster import AgglomerativeClustering   
#import scipy.cluster.hierarchy as shc  
#dendro = shc.dendrogram(shc.linkage(x, method="ward"))  
#plt.title("Dendrogrma Plot")  
#plt.ylabel("Euclidean Distances")  
#plt.xlabel("Customers")
#plt.savefig('Hierarchical_clustering//dendrogram.png', bbox_inches='tight')
#plt.show()
model= AgglomerativeClustering(n_clusters=5, linkage='ward')  
y_predict= model.fit_predict(x) 
plt.scatter(x[:,0], x[:,1], c=y_predict)
plt.title('Clusters of customers')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend()  
plt.savefig('Hierarchical_clustering//cluster.png', bbox_inches='tight')
plt.show()