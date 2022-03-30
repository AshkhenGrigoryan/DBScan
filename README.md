# DBScan

    import numpy as np
    from sklearn.metrics import pairwise_distances

    class DBScan:
    
      def __init__(self, eps = 0.5, minpts = 4):
        self.eps = eps
        self.minpts = minpts
        self.clusters = None
    
      def fit(self, X):
        cluster = 0
        self.clusters = np.array([-1] * len(X))
        self.W = pairwise_distances(X)
        for i, x in enumerate(X):
            if self.label(i, X) == "Noise":
                self.clusters[i] = -2
        while True:
            try:
                i = np.random.choice(np.where(self.clusters == -1)[0])
                if self.isHighDensity(i, X):
                    self.hasbeenselected = set()
                    for index in self.createCluster(i, X):
                        self.clusters[index] = cluster
                    cluster += 1
            except:
                break
    
      def isHighDensity(self, i, X):
        k = 0
        for j, y in enumerate(X):
            if self.W[i, j] < self.eps:
                k += 1
                if k >= self.minpts:
                    return True
        return False
    
      def label(self, i, X):
        if self.isHighDensity(i, X):
            return "Core"
        else:
            for j, y in enumerate(X):
                if self.W[i, j] < self.eps and self.isHighDensity(j, X):
                    return "Border"
            return "Noise"
    
    
      def createCluster(self, i, X):
        cluster = []
        cluster.append(i)
        self.hasbeenselected.add(i)
        for j, y in enumerate(X):
            if self.W[i, j] < self.eps and not (j in self.hasbeenselected):
                cluster.append(j)
                self.hasbeenselected.add(j)
                if self.isHighDensity(j, X):
                    cluster.extend(self.createCluster(j, X))
        return cluster
