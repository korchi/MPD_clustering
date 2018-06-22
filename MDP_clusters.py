import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys
import math
import itertools
import numpy as np
from scipy import linalg
import matplotlib as mpl

from sklearn import mixture
from sklearn import cluster

color_iter = ['blue', 'c', 'green', 'gold','darkorange', 'cornflowerblue', "red", "navy", "magenta", "black"]

def read_data_from_csv(filename):
	data = []
	with open(filename, "r") as csvfile:
		reader = csv.reader(csvfile, delimiter=",")
		for line in reader:
			#reduce to 2D 
			data.append((line[0], int(line[2]), int(line[3]), int(line[4])))
	return data

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
	data = read_data_from_csv(sys.argv[1])
	print("reading complete")
	x = []
	y = []
	size = []
	points = []
	i = 0
	mask = np.random.rand(1, len(data))[0]
	#vizualise
	for (ip, ip_only, port_only, both) in data:
		total_actions= ip_only + port_only + both
		if mask[i] > 0.25 and total_actions>0:
			x.append((ip_only + both)/total_actions)
			y.append((port_only+both)/total_actions)
			size.append(total_actions)
			points.append((x[-1], y[-1]))
		i +=1
	print("Vizualising {} nodes".format(len(x)))
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x, y, alpha=0.2,c=size)
	ax.set_xlabel('Probability of changing IP')
	ax.set_ylabel('Probability of changing Port')
	plt.show()
	"""	

	#K-means++
	#kmeans = cluster.KMeans(n_clusters=5, init="k-means++", n_init=10)
	#res = kmeans.fit_predict(points, y=None)
	
	#GMM
	n_clusters = 5
	gmm = mixture.GaussianMixture(n_components=n_clusters, n_init=30).fit(np.asarray(points))
	res = gmm.predict(np.asarray(points))
	means = gmm.means_
	cov = gmm.covariances_
	print(cov)
	#Spectral clustering
	#spc = cluster.SpectralClustering(n_clusters=5, eigen_solver=None, random_state=None, n_init=3, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1)
	#res = spc.fit_predict(points, y=None)
	population = {}
	colors = []
	for i in range(0,n_clusters):
		population[i] = 0
	for i in res:
		colors.append(color_iter[i])
		population[i] +=1
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(x, y, alpha=0.3,c=colors)
	ax.set_xlabel('Probability of changing IP')
	ax.set_ylabel('Probability of changing Port')
	for k in population.keys():
		print("Cluster {}({}) size={} ({}%), mean={}".format(k,color_iter[k],population[k], population[k]*100/len(points), means[i]))
	plt.show()







