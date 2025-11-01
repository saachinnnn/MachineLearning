# This is the code implementing the KMeans Clustering Algorithm from scratch.
import numpy as np
class Clustering:
    def __init__(self,k : int = 3,max_iterations : int = 100):
        self.k = k  # We store the number of clusters.
        self.max_iterations = max_iterations # The number of maximum allowed iterations during clustering is stored here.
        self.centroids = None # This will store the centroids that we will calculate in the future,i.e the mean of the clusters.
    
    def fit(self,X):
        # Step 1 , we have to assign those random points as centroids initially.
        random_indices = np.random.choice(len(X),self.k,replace = False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iterations):
            # Step 2 to assign the clusters for each of the data points.
            labels = self._assign_clusters(X)
        
            # Step 3 to compute the centroids for each of these clusters by determining their mean(center).
            new_centroid = self._calculate_centroids(X,labels)
        
            # Now we have to check for convergence, which is mainly if the previous clusterd output matches the current, that can be figured out by simply comapring the centroids of the two computations.
            if np.allclose(self.centroids,new_centroid):
                break 
            
            # Update the centroids if convergence is not achieved yet.
            self.centroids = new_centroid
        self.labels = self._assign_clusters(X) # The final labels in the cluster.

    def _assign_clusters(self,X):
        # First compute the distance between the centroid and the data points.
        distances = np.linalg.norm(X[:,np.newaxis] - self.centroids,axis = 2)
        return np.argmin(distances,axis = 1) # This returns whichever cluster each datapoint belongs to.
    
    def _calculate_centroids(self,X,labels):
        new_centroids = [] # This is the new_centroids that we compute.
        for i in range(self.k):
            points_in_cluster = X[labels == i] # Whichever data point lies in that cluster.
            
            if len(points_in_cluster) == 0:
                new_centroids.append(self.centroids[i])
            else:
                new_centroids.append(points_in_cluster.mean(axis = 0))
        return np.array(new_centroids)

    def predict(self,X):
        return self._assign_clusters(X) # The final prediction.

# Now to test this.
X = np.array([
    [1,2],[1,4],[1,0],
    [4,2],[4,4],[4,0]
])
kmeans = Clustering(k = 2)
kmeans.fit(X)

print("Centroids:\n", kmeans.centroids)
print("Labels:", kmeans.labels)
        