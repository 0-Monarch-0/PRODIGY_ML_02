import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#import data
data = pd.read_csv('Mall_Customers.csv')
print(data.head())# gives the names of all the columns
print(data.info())# checking for the null values and data types of the features.
# selecting the features.
x = data[['Annual Income (k$)','Spending Score (1-100)']]

#scaling the features.
scaled = StandardScaler()
x_scaled = scaled.fit_transform(x)

#using elbow methond to dermine the optimal k value to cluster data
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

#plot the results to plot the elbow
plt.plot(range(1,11), wcss)
plt.title('elbow methos')
plt.xlabel('k_values')
plt.ylabel('wcss')
plt.show()

# after checking the graph 3,5 is the optimal for k

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300,n_init=10,random_state=42)
clusters = kmeans.fit_predict(x_scaled)

data['Cluster'] = clusters

# plotting the data

plt.figure(figsize=(8,6))
plt.scatter(x_scaled[:,0], x_scaled[:,1], c=data['Cluster'], cmap = 'viridis')
plt.title("K_MEANS clustering on customer past purchases")
plt.xlabel('Annual income')
plt.ylabel('Spending Score')
plt.show()