import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.title('Visualisation')
plt.show()
#Now we utilize the elbow method to visualize the intertia for different values of K:
data = list(zip(x, y)) #list of data
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

#We can see that the "elbow" on the graph above (where the interia becomes more linear) is at K=2.
# We can then fit our K-means algorithm one more time and plot the different clusters assigned to the data:
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.title('New visualisation, K=2')
plt.show()

