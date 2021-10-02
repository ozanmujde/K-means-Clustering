from math import sqrt  # for Euclidean distance its a necessity


class KMeansClusterClassifier:
    def __init__(self, n_cluster) -> None:
        self.k = n_cluster
        self.epochs = 30
        self.centroids = {}
        self.centPredict = {}
        self.total_euc = 0

    def fit(self, X, y):
        resultClusters = None
        for i in range(self.k):
            # Our data was already shuffled so use random values
            self.centroids[i] = X[i]
        # Klasik Algoritma
        for _ in range(self.epochs):
            clusters = self.calculate_Clusters(X)
            centroids = self.calculate_New_Centroids(clusters)
            resultClusters = clusters
        
        clusterspred = {i: [] for i in range(self.k)}
        #Datanin hangi clustera girecegini anla
        for i in range(len(X)):
            clusterspred[self.cal_True_Cluster(X[i])].append(y[i])
        #Clusterlarda en cok hangi y varsa o centroid'in y labelli o olsun
        for i in range(len(clusterspred)):
            self.centPredict[i] = self.predictAs(clusterspred[i])
        # elbow icin euc
        self.total_euc = 0
        for data in X:
            self.total_euc += self.euc_Distance(
                self.centroids[self.cal_True_Cluster(data)], data)

        self.total_euc /= (len(X))

        print( self.total_euc)
        print(self.centPredict)

        return self.centroids, self.total_euc, resultClusters

    def predict(self, X):
        clusters = {i: [] for i in range(self.k)}
        predY = [0 for i in range(len(X))]
        for i in range(len(X)):
            index = self.cal_True_Cluster(X[i])
            clusters[index].append(X[i])
            predY[i] = self.centPredict[index]
        return predY

    def euc_Distance(self, centroid, data):
        return sqrt(sum((centroid[i] - data[i])**2 for i in range(len(centroid))))

    def calculate_Clusters(self, X):
        clusters = {i: [] for i in range(self.k)}
        for data in X:
            euc_dist = [self.euc_Distance(self.centroids[j],
                                          data) for j in range(self.k)]
            clusters[euc_dist.index(min(euc_dist))].append(data)
        return clusters

    def calculate_New_Centroids(self, clusters):
        for i in range(self.k):
            #Verinin ortasina koy
            self.centroids[i] = self.average(clusters[i])
        return self.centroids

    def minIndex(self, euc_dist):
        index = -1
        min = 2147483647  # max int without implement library
        for i in range(len(euc_dist)):
            if euc_dist[i] < min:
                index = i
                min = euc_dist[i]
        return index

    def average(self, array):
        arr = [0 for i in range(len(array[0]))]
        for j in range(len(array[0])):
            total = sum(array[i][j] for i in range(len(array)))
            arr[j] = total / len(array)
        return arr

    def cal_True_Cluster(self, element):
        min = 1000
        Index = -1
        for i in range(len(self.centroids)):
            if self.euc_Distance(self.centroids[i], element) < min:
                min = self.euc_Distance(self.centroids[i], element)
                Index = i
        return Index

    def predictAs(self, cluster):
        count0 = 0
        count1 = 0
        count2 = 0
        for y in cluster:
            if y == 0:
                count0 += 1
            elif y == 1:
                count1 += 1
            elif y == 2:
                count2 += 1
        if count0 >= count1:
            if count0 >= count2:
                return 0
            else:
                return 2
        elif count1 >= count2:
            return 1
        else:
            return 2

    def transformY(self, Y):
        #Y yi int e cevir
        length = len(Y)
        arr = [0 for i in range(length)]
        for i in range(length):
            if Y[i] == 'Iris-setosa':
                arr[i] = 0
            elif Y[i] == 'Iris-versicolor':
                arr[i] = 1
            elif Y[i] == 'Iris-virginica':
                arr[i] = 2
        return arr
