import numpy as np
import torch



# K-means Model
class KmeansModel():
    def __init__(self, n_clusters=None, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels_ = None
        self.distances = None

    def fit(self, X):

        self.centers = torch.nn.Parameter(
            X[np.random.choice(len(X), self.n_clusters, replace=False)])

        prev_centers = self.centers.data.clone()
        for i in range(self.max_iter):
            distances = torch.cdist(X, self.centers)
            _, cluster_assignments = torch.min(distances, dim=-1)
            self._update_centers(X, cluster_assignments)
            if self._has_converged(prev_centers):
                print(f"Converged after {i + 1} iterations.")
                break
            prev_centers = self.centers.data.clone()
        self.labels_ = torch.min(distances, dim=-1).indices
        self.distances = distances
        return self

    def _update_centers(self, X,cluster_assignments):
        for i in range(self.centers.size(0)):
            cluster_points = X[cluster_assignments == i]
            if len(cluster_points) > 0:
                self.centers.data[i] = torch.mean(cluster_points, dim=0).cpu()

    def _has_converged(self, prev_centers):
        return torch.equal(prev_centers, self.centers.data)



# K-means++ Model
class KMeansModel_plusplus():
    def __init__(self, n_clusters=None, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels_ = None
        self.distances = None

    def fit(self, X):
        self.centers = torch.nn.Parameter(self.kmeans_plusplus_init(X, self.n_clusters))
        prev_centers = self.centers.data.clone()

        for i in range(self.max_iter):

            distances = torch.cdist(X, self.centers)
            _, cluster_assignments = torch.min(distances, dim=-1)


            self._update_centers(X, cluster_assignments)

            if self._has_converged(prev_centers):
                print(f"Converged after {i + 1} iterations.")
                break

            prev_centers = self.centers.data.clone()

        self.labels_ = torch.min(distances, dim=-1).indices
        self.distances = distances
        return self

    def kmeans_plusplus_init(self, tensor, n_clusters):
        n_samples, _ = tensor.shape
        initial_center_idx = np.random.choice(n_samples)
        centers = [tensor[initial_center_idx]]


        for _ in range(1, n_clusters):
            distances = torch.cdist(tensor, torch.stack(centers), p=2)
            min_distances = distances.min(dim=1).values
            probabilities = min_distances / min_distances.sum()
            next_center_idx = torch.multinomial(probabilities, 1).item()
            centers.append(tensor[next_center_idx])

        return torch.stack(centers)

    def _update_centers(self, X, cluster_assignments):
        for i in range(self.centers.size(0)):
            cluster_points = X[cluster_assignments == i]
            if len(cluster_points) > 0:
                self.centers.data[i] = torch.mean(cluster_points, dim=0).cpu()

    def _has_converged(self, prev_centers):
        return torch.equal(prev_centers, self.centers.data)




# Mean Shift Model
class MeanShiftModel():
    def __init__(self, bandwidth=0.5, max_iter=300, tol=1e-3):
        self.n_clusters = None
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.centers = None
        self.labels_ = None
        self.distances = None

    def fit(self, X):
        self.bandwidth = self.estimate_bandwidth(X)
        n_samples, n_features = X.shape
        centers = X.clone()

        for _ in range(self.max_iter):
            distances = torch.cdist(X, centers)
            new_centers = []
            for i in range(n_samples):
                within_bandwidth = distances[i] < self.bandwidth
                if torch.sum(within_bandwidth) > 0:
                    new_center = X[within_bandwidth].mean(dim=0)
                else:
                    new_center = centers[i]
                new_centers.append(new_center)

            new_centers = torch.stack(new_centers)
            shift = torch.norm(new_centers - centers, dim=1).max()
            centers = new_centers
            if shift < self.tol:
                break

        self.centers = centers
        self.labels_ = torch.min(distances, dim=-1).indices
        self.distances = distances
        self.n_clusters = torch.unique(self.labels_).numel()
        return self

    # A method that adaptively adjusts bandwidth, with the quantile value between 0 and 1, to control the number of clusters after the final clustering
    def estimate_bandwidth(self,X, quantile=0.3, n_samples=None, random_state=None):
        if random_state is not None:
            torch.manual_seed(random_state)

        if n_samples is not None and n_samples < X.size(0):
            indices = torch.randperm(X.size(0))[:n_samples]
            X = X[indices]

        n_neighbors = max(1, int(X.size(0) * quantile))
        distances = torch.cdist(X, X, p=2)
        distances.fill_diagonal_(float('inf'))
        max_distances = torch.topk(distances, n_neighbors, largest=False, dim=1).values[:, -1]
        bandwidth = torch.mean(max_distances).item()
        return bandwidth
