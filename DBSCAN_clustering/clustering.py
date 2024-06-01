import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.cluster import DBSCAN
dataset = pd.read_csv('DBSCAN_clustering//Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values
model = DBSCAN(eps=0.3, min_samples=25)
model.fit(x)
y_prediction = model.fit_predict(x)
plt.scatter(x[:,0], x[:,1], c=y_prediction)
plt.title('Clusters of customers')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend()  
plt.savefig('DBSCAN_clustering//cluster.png', bbox_inches='tight')
plt.show()